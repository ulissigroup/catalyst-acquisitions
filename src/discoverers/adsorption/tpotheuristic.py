'''
This submodule houses the `TpotHeuristic` child class of
`AdsorptionDiscovererBase` that hallucinates the performance of an incumbent
method that uses a TPOT-defined model to perform a sort of Thompson sampling in
the context of discovering catalysts by screening their adsorption energies.

Refer to https://www.nature.com/articles/s41929-018-0142-1 for more details.
'''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'


import os
import gc
import random
import pickle
from pathlib import Path
from bisect import bisect_right
import numpy as np
from scipy.stats import norm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tpot import TPOTRegressor
from gaspy.gasdb import get_surface_from_doc
from gaspy_regress import fingerprinters
from .adsorption_base import AdsorptionDiscovererBase


class TpotHeuristic(AdsorptionDiscovererBase):
    '''
    This discoverer uses a Gaussian selection method with a TPOT model to select
    new sampling points.

    ...sorry for the awful code. This is a hack-job and I know it.
    '''

    def __init__(self, *args, **kwargs):
        '''
        In addition to the normal things that this class's parent classes do in
        `__init__` this method also instantiates the `TPOTWrapper`
        '''
        self.assumed_stdev = 0.1
        self.model = TPOTWrapper()
        Path(self.cache_location).mkdir(exist_ok=True)
        super().__init__(*args, **kwargs)

    @property
    def cache_location(self):
        return './tpot_caches/'

    def _train(self, next_batch):
        '''
        Calculate the residuals of the current training batch, then retrain on
        everything

        Args:
            next_batch  The output of this class's `_choose_next_batch` method
        '''
        # Parse the incoming batch
        try:
            features, dft_energies, next_surfaces = next_batch
        except ValueError:
            features, dft_energies = next_batch
            next_surfaces = [get_surface_from_doc(doc) for doc in features]

        # Calculate and save the results of this next batch
        try:
            tpot_predictions, uncertainties = self.model.predict(features)
            residuals = tpot_predictions - dft_energies
            self.uncertainties.extend(uncertainties)
            self.residuals.extend(residuals.tolist())
        # If prediction doesn't work, then we probably haven't trained the
        # first batch. And if haven't done this, then there's no need to save
        # the residuals and uncertainty estimates.
        except AttributeError:
            pass

        # Retrain
        self.training_features.extend(features)
        self.training_labels.extend(dft_energies)
        self.training_surfaces.extend(next_surfaces)
        self.model.train(self.training_features, self.training_labels)
        self._save_current_run()

    def _choose_next_batch(self):
        '''
        Choose the next batch "randomly", where the probability of selecting
        sites are weighted using a combination of a Gaussian distribution and
        TPOT's prediction of their distance from the optimal energy.

        Returns:
            features    The features that this method chose to investigate next
            labels      The labels that this method chose to investigate next
            surfaces    The surfaces that this method chose to investigate next
        '''
        # Use the energies to calculate probabilities of selecting each site
        energies, _ = self.model.predict(self.sampling_features)
        gaussian_distribution = norm(loc=self.target_energy, scale=self.assumed_stdev)
        probability_densities = [gaussian_distribution.pdf(energy) for energy in energies]

        # Perform a weighted shuffling of the sampling space such that sites
        # with better energies are more likely to be early in the list
        features, labels, surfaces = self.weighted_shuffle(self.sampling_features,
                                                           self.sampling_labels,
                                                           self.sampling_surfaces,
                                                           weights=probability_densities)
        self.sampling_features = features
        self.sampling_labels = labels
        self.sampling_surfaces = surfaces

        # Now that the samples are sorted, find the next ones and add them to
        # the training set
        features, labels, surfaces = self._pop_next_batch()
        self.training_features.extend(features)
        self.training_labels.extend(labels)
        self.training_surfaces.extend(surfaces)
        return features, labels, surfaces

    @staticmethod
    def weighted_shuffle(*sequences, weights):
        '''
        This function will shuffle a sequence using weights to increase the chances
        of putting higher-weighted elements earlier in the list. Credit goes to
        Nicky Van Foreest, whose function I based this off of.

        Args:
            sequence    Any number of sequences of elements that you want
                        shuffled
            weights     A sequence that is the same length as the `sequence`
                        that contains the corresponding probability weights for
                        selecting/choosing each element in `sequence`
        Returns:
            shuffled_list   A list whose elements are identical to those in the
                            `sequence` argument, but randomly shuffled such
                            that the elements with higher weights are more
                            likely to be in the front/start of the list.
        '''
        shuffled_arrays = [np.empty_like(sequence) for sequence in sequences]

        # Pack together the elements in the sequences and their respective weights
        packets = list(zip(*sequences, weights))
        for i in range(len(packets)):

            # Randomly choose one of the elements, and get the corresponding index
            cumulative_weights = np.cumsum([packet[-1] for packet in packets])
            rand = random.random() * cumulative_weights[-1]
            selected_index = bisect_right(cumulative_weights, rand)

            # Pop the element out so we don't re-select
            packet = packets.pop(selected_index-1)

            # Don't need to save the last item in each packet, which is the
            # weight
            for j, value in enumerate(list(packet)[:-1]):
                shuffled_arrays[j][i] = value

        return (array.tolist() for array in shuffled_arrays)


class TPOTWrapper:
    '''
    This is our wrapper for fingerprinting sites and then using TPOT to predict
    adsorption energies from those fingerprints.
    '''
    def __init__(self):
        '''
        Instantiate the preprocessing pipeline and the TPOT model
        '''
        # Instantiate the fingerprinter
        inner_fingerprinter = fingerprinters.InnerShellFingerprinter()
        outer_fingerprinter = fingerprinters.OuterShellFingerprinter()
        fingerprinter = fingerprinters.StackedFingerprinter(inner_fingerprinter,
                                                            outer_fingerprinter)
        scaler = StandardScaler()
        pca = PCA()
        preprocessing_pipeline = Pipeline([('fingerprinter', fingerprinter),
                                           ('scaler', scaler),
                                           ('pca', pca)])
        self.preprocessor = preprocessing_pipeline

        # Instantiate TPOT
        self.tpot = TPOTRegressor(generations=2,
                                  population_size=32,
                                  offspring_size=32,
                                  verbosity=2,
                                  scoring='neg_median_absolute_error',
                                  n_jobs=16,
                                  warm_start=True)

    def train(self, docs, energies):
        '''
        Trains both the preprocessor and TPOT in series

        Args:
            docs        List of dictionaries from
                        `gaspy.gasdb.get_adsorption_docs`
            energies    List of floats containing the adsorption energies of
                        `docs`
        '''
        features = self.preprocessor.fit_transform(docs)
        self.tpot.fit(features, energies)

        # Try to address some memory issues by collecting garbage
        _ = gc.collect()  # noqa: F841

    def predict(self, docs):
        '''
        Use the whole fingerprinting and TPOT pipeline to make adsorption
        energy predictions

        Args:
            docs        List of dictionaries from
                        `gaspy.gasdb.get_adsorption_docs`
        Returns:
            predictions     `np.array` of TPOT's predictions of each doc
            uncertainties   `np.array` that contains the "uncertainty
                            prediction" for each site. In this case, it'll
                            just be TPOT's RMSE
        '''
        # Point predictions
        features = self.preprocessor.transform(docs)
        predictions = np.array(self.tpot.predict(features))

        # "Uncertainties" will just be the RMSE
        residuals = np.array([prediction - doc['energy']
                              for prediction, doc in zip(predictions, docs)])
        rmse = np.sqrt((residuals**2).mean())
        uncertainties = np.array([rmse for _ in predictions])

        return predictions, uncertainties

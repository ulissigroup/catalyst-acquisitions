'''
This submodule houses the `TpotThompson` child class of
`AdsorptionDiscovererBase` that hallucinates the performance of an incumbent
method that uses a TPOT-defined model to perform a sort of Thompson sampling in
the context of discovering catalysts by screening their adsorption energies.

Refer to https://www.nature.com/articles/s41929-018-0142-1 for more details.
'''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'


import gc
import random
import pickle
from bisect import bisect_right
import numpy as np
from scipy.stats import norm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tpot import TPOTRegressor
from gaspy_regress import fingerprinters
from .benchmarks import AdsorptionDiscovererBase


class TpotThompson(AdsorptionDiscovererBase):
    '''
    This discoverer uses a Gaussian selection method with a TPOT model to select
    new sampling points.

    ...sorry for the awful code. This is a hack-job and I know it.
    '''
    # The width of the Gaussian selection curve
    assumed_stdev = 0.1

    def _train(self):
        '''
        Calculate the residuals of the current training batch, then retrain on
        everything
        '''
        # Instantiate the preprocessor and TPOT if we haven't done so already
        if not hasattr(self, 'preprocessor'):
            self._train_preprocessor()
        if not hasattr(self, 'tpot'):
            self.tpot = TPOTRegressor(generations=2,
                                      population_size=32,
                                      offspring_size=32,
                                      verbosity=2,
                                      scoring='neg_median_absolute_error',
                                      n_jobs=16,
                                      warm_start=True)
            features = self.preprocessor.transform(self.training_batch)
            energies = [doc['energy'] for doc in self.training_batch]
            self.tpot.fit(features, energies)

        # Calculate and save the residuals of this next batch
        features = self.preprocessor.transform(self.training_batch)
        tpot_predictions = self.tpot.predict(features)
        dft_energies = np.array([doc['energy'] for doc in self.training_batch])
        residuals = tpot_predictions - dft_energies
        self.residuals.extend(list(residuals))

        # Retrain
        self.training_set.extend(self.training_batch)
        self.__train_tpot()

    def _train_preprocessor(self):
        '''
        Trains the preprocessing pipeline and assigns it to the `preprocessor`
        attribute.
        '''
        inner_fingerprinter = fingerprinters.InnerShellFingerprinter()
        outer_fingerprinter = fingerprinters.OuterShellFingerprinter()
        fingerprinter = fingerprinters.StackedFingerprinter(inner_fingerprinter,
                                                            outer_fingerprinter)
        scaler = StandardScaler()
        pca = PCA()
        preprocessing_pipeline = Pipeline([('fingerprinter', fingerprinter),
                                           ('scaler', scaler),
                                           ('pca', pca)])
        preprocessing_pipeline.fit(self.training_batch)
        self.preprocessor = preprocessing_pipeline

    def __train_tpot(self):
        '''
        Train TPOT using the `training_set` attached to the class
        '''
        # Cache the current point for (manual) warm-starts, because there's a
        # solid chance that TPOT might cause a segmentation fault.
        cache_name = 'caches/%.3i_discovery_cache.pkl' % self.next_batch_number
        with open(cache_name, 'wb') as file_handle:
            cache = {'training_features': self.training_features,
                     'training_labels': self.training_labels,
                     'training_surfaces': self.training_surfaces,
                     'sampling_features': self.sampling_features,
                     'sampling_labels': self.sampling_labels,
                     'sampling_surfaces': self.sampling_surfaces,
                     'batch_size': self.batch_size,
                     'residuals': self.residuals,
                     'uncertainties': self.uncertainties,
                     'reward_history': self.reward_history,
                     'next_batch_number': self.next_batch_number}
            pickle.dump(cache, file_handle)

        # Instantiate the preprocessor and TPOT if we haven't done so already
        self._train_preprocessor()
        if not hasattr(self, 'tpot'):
            self.tpot = TPOTRegressor(generations=2,
                                      population_size=32,
                                      offspring_size=32,
                                      verbosity=2,
                                      scoring='neg_median_absolute_error',
                                      n_jobs=16,
                                      warm_start=True)

        # [Re-]train
        features = self.preprocessor.transform(self.training_features)
        energies = [doc['energy'] for doc in self.training_labels]
        self.tpot.fit(features, energies)
        self.next_batch_number += 1

        # Try to address some memory issues by collecting garbage
        _ = gc.collect()  # noqa: F841

    def _choose_next_batch(self):
        '''
        Choose the next batch "randomly", where the probability of selecting
        sites are weighted using a combination of a Gaussian distribution and
        TPOT's prediction of their distance from the optimal energy.
        '''
        # Use the energies to calculate probabilities of selecting each site
        features = self.preprocessor.transform(self.sampling_features)
        energies = self.tpot.predict(features)
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
            packet = packets.pop(selected_index)

            # Don't need to save the last item in each packet, which is the
            # weight
            for j, value in enumerate(list(packet)[:-1]):
                shuffled_arrays[j][i] = value

        return (array.tolist() for array in shuffled_arrays)

    def _pop_next_batch(self):
        '''
        Optional helper function that you can use to choose the next batch from
        `self.sampling_features`, remove it from the attribute, place the new
        batch onto the `self.training_features` attribute, increment the
        `self.next_batch_number`. Then do it all again for the
        `self.sampling_labels` and `self.training_labels` attributes.

        This method will only work if you have already sorted the
        `self.sampling_features` and `self.sampling_labels` such that the
        highest priority samples are earlier in the index.

        Returns:
            features    A list of length `self.batch_size` that contains the
                        next batch of features to train on.
            labels      A list of length `self.batch_size` that contains the
                        next batch of labels to train on.
        '''
        features = []
        labels = []
        surfaces = []
        for _ in range(self.batch_size):
            try:
                feature = self.sampling_features.pop(0)
                label = self.sampling_labels.pop(0)
                surface = self.sampling_surfaces.pop(0)
                features.append(feature)
                labels.append(label)
                surfaces.append(surface)
            except IndexError:
                break
        self.next_batch_number += 1
        return features, labels, surfaces

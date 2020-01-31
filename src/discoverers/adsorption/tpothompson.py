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
    stdev = 0.1

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
        # Open the cached preprocessor
        try:
            cache_name = 'caches/preprocessor.pkl'
            with open(cache_name, 'rb') as file_handle:
                self.preprocessor = pickle.load(file_handle)

        # If there is no cache, then remake it
        except FileNotFoundError:
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

            # Cache it for next time
            with open(cache_name, 'wb') as file_handle:
                pickle.dump(preprocessing_pipeline, file_handle)

    def __train_tpot(self):
        '''
        Train TPOT using the `training_set` attached to the class
        '''
        # Cache the current point for (manual) warm-starts, because there's a
        # solid chance that TPOT might cause a segmentation fault.
        cache_name = 'caches/%.3i_discovery_cache.pkl' % self.next_batch_number
        with open(cache_name, 'wb') as file_handle:
            cache = {'training_set': self.training_set,
                     'sampling_space': self.sampling_space,
                     'residuals': self.residuals,
                     'regret_history': self.regret_history,
                     'next_batch_number': self.next_batch_number,
                     'training_batch': self.training_batch}
            pickle.dump(cache, file_handle)

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

        # [Re-]train
        features = self.preprocessor.transform(self.training_set)
        energies = [doc['energy'] for doc in self.training_set]
        self.tpot.fit(features, energies)
        self.next_batch_number += 1

        # Try to address some memory issues by collecting garbage
        _ = gc.collect()  # noqa: F841

    def _choose_next_batch(self):
        '''
        Choose the next batch "randomly", where the probability of selecting
        sites are weighted using a combination of a Gaussian distribution and
        TPOT's prediction of their distance from the optimal energy. Snippets
        were stolen from the GASpy_feedback module.
        '''
        # Use the energies to calculate probabilities of selecting each site
        features = self.preprocessor.transform(self.sampling_space)
        energies = self.tpot.predict(features)
        gaussian_distribution = norm(loc=self.optimal_value, scale=self.stdev)
        probability_densities = [gaussian_distribution.pdf(energy) for energy in energies]

        # Perform a weighted shuffling of the sampling space such that sites
        # with better energies are more likely to be early in the list
        self.sampling_space = self.weighted_shuffle(self.sampling_space,
                                                    probability_densities)

        self._pop_next_batch

    @staticmethod
    def weighted_shuffle(sequence, weights):
        '''
        This function will shuffle a sequence using weights to increase the chances
        of putting higher-weighted elements earlier in the list. Credit goes to
        Nicky Van Foreest, whose function I based this off of.

        Args:
            sequence    A sequence of elements that you want shuffled
            weights     A sequence that is the same length as the `sequence` that
                        contains the corresponding probability weights for
                        selecting/choosing each element in `sequence`
        Returns:
            shuffled_list   A list whose elements are identical to those in the
                            `sequence` argument, but randomly shuffled such that
                            the elements with higher weights are more likely to
                            be in the front/start of the list.
        '''
        shuffled_list = np.empty_like(sequence)

        # Pack the elements in the sequences and their respective weights
        pairings = list(zip(sequence, weights))
        for i in range(len(pairings)):

            # Randomly choose one of the elements, and get the corresponding index
            cumulative_weights = np.cumsum([weight for _, weight in pairings])
            rand = random.random() * cumulative_weights[-1]
            j = bisect_right(cumulative_weights, rand)

            # Pop the element out so we don't re-select
            try:
                shuffled_list[i], _ = pairings.pop(j)

            # Hack a quick fix to some errors I don't feel like solving
            except IndexError:
                try:
                    shuffled_list[i], _ = pairings.pop(-1)
                except IndexError:
                    break

        return shuffled_list.tolist()

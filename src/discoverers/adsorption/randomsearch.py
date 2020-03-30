'''
This submodule houses the `RandomSearcher` child class of
`AdsorptionDiscovererBase` that hallucinates the performance of random search
in the context of discovering catalysts by screening their adsorption energies.
'''

__author__ = 'Willie Neiswanger'
__email__ = 'willie@cs.cmu.edu'


import warnings
import random
from .benchmarks import AdsorptionDiscovererBase

# The tqdm autonotebook is still experimental, and it warns us. We don't care,
# and would rather not hear about the warning everytime.
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from tqdm.autonotebook import tqdm


class RandomSearcher(AdsorptionDiscovererBase):
    '''
    This discoverer carries out a random search procedure to find adsorption
    energies.
    '''

    def _train(self):
        '''
        While random search does not technically use a model, we must still
        implement this method.
        '''

        # NOTE: We could do something like compute random predictions (or
        # shuffle predictions in the training set) in order to get some sort of
        # proxy set of residuals. This is a bit difficult to do correctly,
        # especially if the self.training_batch is small.

        # Mandatory extension of the training set to include this next batch
        self.training_set.extend(self.training_batch)

    def _choose_next_batch(self):
        '''
        Choose the next batch uniformly at random.
        '''
        self.__shuffle_sampling_space()

        features, labels, surfaces = self._pop_next_batch()
        self.training_features.extend(features)
        self.training_labels.extend(labels)
        self.training_surfaces.extend(surfaces)
        return features, labels, surfaces

    def __shuffle_sampling_space(self):
        '''
        Randomly shuffle self.sampling_features and self.sampling_labels.
        '''
        sampling_all = list(zip(self.sampling_features, self.sampling_labels))
        random.shuffle(sampling_all)
        self.sampling_features, self.sampling_labels = zip(*sampling_all)

        ### TODO: shuffle surfaces too?

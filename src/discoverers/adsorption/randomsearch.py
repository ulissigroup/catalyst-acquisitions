'''
This submodule houses the `NullModelRandomSearcher` child class of
`BaseAdsorptionDiscoverer` that hallucinates the performance of random search
under a null model (which always predicts zeros with fixed uncertainties) in
the context of discovering catalysts by screening their adsorption energies.
'''

import random
from .base import BaseAdsorptionDiscoverer


class RandomSearcher(BaseAdsorptionDiscoverer):
    '''
    This discoverer carries out a random search procedure, under a null model,
    to find adsorption energies.
    '''

    def _choose_next_batch(self):
        ''' Choose the next batch uniformly at random '''
        self.__shuffle_sampling_space()
        features, labels, surfaces = self._pop_next_batch()
        return features, labels, surfaces

    def __shuffle_sampling_space(self):
        '''
        Randomly shuffle self.sampling_features and self.sampling_labels.
        '''
        sampling_all = list(zip(self.sampling_features,
                                self.sampling_labels,
                                self.sampling_surfaces))
        random.shuffle(sampling_all)
        sampling_features, sampling_labels, sampling_surfaces = zip(*sampling_all)
        self.sampling_features = list(sampling_features)
        self.sampling_labels = list(sampling_labels)
        self.sampling_surfaces = list(sampling_surfaces)

'''
This submodule uses the same CFGP model used in the MMS method, but instead of
using MMS to sample the sites, it samples them completely at random. It
provides a baseline against which to compare the acquisition function.
'''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'


import random
from .mms import MultiscaleDiscoverer


class RandomSearcherCFGP(MultiscaleDiscoverer):
    '''
    This discoverer carries out a random search procedure to find adsorption
    energies. It then trains a CFGP on the data to predict bulk values.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def cache_location(self):
        return './rs_caches/'

    def _choose_next_batch(self):
        ''' Choose the next batch uniformly at random. '''
        self.__shuffle_sampling_space()
        features, labels, _ = self._pop_next_batch()
        return features, labels

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

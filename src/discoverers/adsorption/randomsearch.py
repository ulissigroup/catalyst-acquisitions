'''
This submodule houses the `RandomSearcher` child class of
`AdsorptionDiscovererBase` that hallucinates the performance of random search
in the context of discovering catalysts by screening their adsorption energies.
'''

__author__ = 'Willie Neiswanger'
__email__ = 'willie@cs.cmu.edu'


import warnings
import random
import numpy as np
from .adsorption_base import AdsorptionDiscovererBase

# The tqdm autonotebook is still experimental, and it warns us. We don't mind,
# and would rather not hear about the warning everytime.
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from tqdm.autonotebook import tqdm


class RandomSearcher(AdsorptionDiscovererBase):
    '''
    This discoverer carries out a random search procedure to find adsorption
    energies.
    '''

    def __init__(self, *args, **kwargs):
        '''
        Instantiate `NullModel`.
        '''
        self.model = NullModel()
        super().__init__(*args, **kwargs)

    def _train(self, next_batch):
        '''
        While random search does not technically use a model, we must still
        implement this method.
        '''

        features, dft_energies, next_surfaces = next_batch

        # Get predictions and uncertainties from NullModel
        predictions, uncertainties = self.model.predict(features)
        residuals = predictions - dft_energies
        self.uncertainties.extend(uncertainties)
        self.residuals.extend(residuals.tolist())

        # Extend training set attributes to include this next batch
        self.training_features.extend(features)
        self.training_labels.extend(dft_energies)
        self.training_surfaces.extend(next_surfaces)

    def _choose_next_batch(self):
        '''
        Choose the next batch uniformly at random.
        '''
        self.__shuffle_sampling_space()
        features, labels, surfaces = self._pop_next_batch()
        return features, labels, surfaces

    def __shuffle_sampling_space(self):
        '''
        Randomly shuffle self.sampling_features and self.sampling_labels.
        '''
        sampling_all = list(zip(self.sampling_features, self.sampling_labels))
        random.shuffle(sampling_all)
        self.sampling_features, self.sampling_labels = zip(*sampling_all)

        ### TODO: shuffle surfaces too?


class NullModel:
    '''
    This is a null model, which does nothing during training, and always
    predicts 0 for mean and 1 for uncertainty.
    '''

    def train(self, docs, energies):
        '''Do nothing.'''
        pass

    def predict(self, docs):
        '''For each doc, predict 0 for mean and 1 for uncertainty.'''
        predictions = np.zeros(len(docs))
        uncertainties = np.ones(len(docs))
        return predictions, uncertainties

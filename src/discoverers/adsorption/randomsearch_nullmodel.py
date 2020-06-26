'''
This submodule houses the `NullModelRandomSearcher` child class of
`AdsorptionDiscovererBase` that hallucinates the performance of random search
under a null model (which always predicts zeros with fixed uncertainties) in
the context of discovering catalysts by screening their adsorption energies.
'''


import warnings
import random
import numpy as np
from .adsorption_base import AdsorptionDiscovererBase

# The tqdm autonotebook is still experimental, and it warns us. We don't mind,
# and would rather not hear about the warning everytime.
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from tqdm.autonotebook import tqdm


class RandomSearcherNullModel(AdsorptionDiscovererBase):
    '''
    This discoverer carries out a random search procedure, under a null model,
    to find adsorption energies.
    '''

    def __init__(self, *args, **kwargs):
        '''
        Instantiate `NullModel`.
        '''
        self.model = NullModel()
        super().__init__(*args, **kwargs)

    @property
    def cache_location(self):
        return './rs_caches/'

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

    def _train(self, next_batch):
        '''
        This function trains the null model (which involves no computation),
        where the null model always predicts a constant value and uncertainty.
        '''

        features, dft_energies, next_surfaces = next_batch

        # Get predictions and uncertainties from NullModel for this next batch
        try:
            predictions, uncertainties = self.model.predict(features)
            residuals = predictions - dft_energies
            self.residuals.extend(residuals.tolist())
            self.uncertainties.extend(uncertainties)
        # If prediction doesn't work, then we probably haven't trained the
        # first batch. And if haven't done this, then there's no need to save
        # the residuals and uncertainty estimates.
        except AttributeError:
            pass

        # Retrain
        self.training_features.extend(features)
        self.training_labels.extend(dft_energies)
        self.training_surfaces.extend(next_surfaces)

        self.model.train()
        self._save_current_run()

    def _save_current_run(self):
        ''' Save the state of the discoverer '''
        super()._save_current_run()

    def load_last_run(self):
        ''' Load the last state of the hallucination '''
        # Load last hallucination state
        super().load_last_run()


class NullModel:
    '''
    This is a null model, which does nothing during training, and always
    predicts 0 for mean and 1 for uncertainty.
    '''

    def train(self):
        '''Do nothing.'''
        pass

    def predict(self, features):
        '''For each doc, predict 0 for mean and 1 for uncertainty.'''
        predictions = np.zeros(len(features))
        uncertainties = np.ones(len(features))
        return predictions, uncertainties

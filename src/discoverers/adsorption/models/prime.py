__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'


import numpy as np
from scipy import stats
from ocpmodels.datasets.gasdb import Gasdb
from .base import BaseModel


class PrimeModel(BaseModel):
    '''
    This is a prime model, which does predicts the correct answer every time
    with an uncertainty sampled from a given chi-squared distribution
    '''

    def __init__(self, db_dir, uncertainty=0.1):
        self.db_dir = db_dir
        self.std_dist = stats.chi2(df=1, loc=0, scale=uncertainty)
        self.dataset = Gasdb({'src': self.db_dir})

    def train(self, _features=None, _labels=None):
        pass

    def predict(self, indices):
        '''
        Return the real energies as the predictions and an uncertainty sampled
        from a chi-squared distribution.
        '''
        predictions = []
        for idx in indices:
            row = list(self.dataset.ase_db.select(idx))[0]
            energy = row.data['adsorption_energy']
            predictions.append(energy)
        predictions = np.array(predictions)

        uncertainties = self.std_dist.rvs(len(indices))
        return predictions, uncertainties

    def save(self):
        pass

    def load(self):
        pass

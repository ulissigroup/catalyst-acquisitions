__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'


import os
import warnings
import numpy as np
from scipy import stats
import ase.db
from ocpmodels.datasets.gasdb import Gasdb
from .base import BaseModel

# The tqdm autonotebook is still experimental, and it warns us. We don't care,
# and would rather not hear about the warning everytime.
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from tqdm.autonotebook import tqdm


class PrimeModel(BaseModel):
    '''
    This is a prime model, which does predicts the correct answer every time
    with an uncertainty sampled from a given chi-squared distribution
    '''

    def __init__(self, db_dir, uncertainty=0.1, df=1):
        '''
        Args:
            db_dir          A string indicating the location of the ASE db
            uncertainty     The scale parameter used in the Chi-Squared
                            distribution that we sample the uncertainties from.
            df              The degrees of freedom used in the Chi-Squared
                            distribution that we sample the uncertainties from.
        '''
        self.db_dir = db_dir
        self.std_dist = stats.chi2(df=df, loc=0, scale=uncertainty)

        # Read from the ASE database once for future speedup
        iterator = tqdm(self.ase_db.select(),
                        desc='reading ASE db for Prime model',
                        total=self.ase_db.count())
        rows = list(iterator)
        self.data_dict = {row.id: row.data['adsorption_energy']
                          for row in tqdm(rows, desc='Reading data')}

    @property
    def ase_db(self):
        '''
        This method/property will use the first `*.db` object in the source
        directory.
        '''
        for file_ in os.listdir(self.db_dir):
            if file_.endswith(".db"):
                raw_file_name = os.path.join(self.db_dir, file_)
                db = ase.db.connect(raw_file_name)
                return db

    def train(self, _features=None, _labels=None):
        pass

    def predict(self, indices):
        '''
        Return the real energies as the predictions and an uncertainty sampled
        from a chi-squared distribution.
        '''
        predictions = np.array([self.data_dict[idx] for idx in indices])
        uncertainties = self.std_dist.rvs(len(indices))
        return predictions, uncertainties

    def save(self):
        pass

    def load(self):
        pass

__authors__ = ['Willie Neiswanger', 'Kevin Tran']
__emails__ = ['ktran@andrew.cmu.edu', 'willie@cs.cmu.edu']


import os
import numpy as np
import ase.db
from ocpmodels.datasets.gasdb import Gasdb
from .base import BaseModel


class NullModel(BaseModel):
    '''
    This is a null model, which does nothing during training, and always
    predicts 0 for mean and 1 for uncertainty.
    '''

    def __init__(self, db_dir, prediction=0.):
        '''
        Save the dataset because MMS will assume that we have it

        Args:
            db_dir      A string indicating the location of the ASE db. Not
                        really used at all. We keep it here to make it
                        syntactically consistent with the other models.
            prediction  A float indicating the static prediction you want this
                        null model to make everytime
        '''
        self.db_dir = db_dir
        self.prediction = prediction


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

    def train(self, *_args):
        pass

    def predict(self, features):
        '''For each doc, predict 0 for mean and 1 for uncertainty.'''
        predictions = np.ones(len(features)) * self.prediction
        uncertainties = np.ones(len(features))
        return predictions, uncertainties

    def save(self):
        pass

    def load(self):
        pass

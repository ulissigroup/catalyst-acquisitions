__authors__ = ['Willie Neiswanger', 'Kevin Tran']
__emails__ = ['ktran@andrew.cmu.edu', 'willie@cs.cmu.edu']


import numpy as np
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
            db_dir      A string indicating the location of the ASE db
            prediction  A float indicating the static prediction you want this
                        null model to make everytime
        '''
        # The frameworks we'll be using assume that we have a dataset attribute
        self.db_dir = db_dir
        self.dataset = Gasdb({'src': self.db_dir})

        self.prediction = prediction

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

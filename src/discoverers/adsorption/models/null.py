__authors__ = ['Willie Neiswanger', 'Kevin Tran']
__emails__ = ['ktran@andrew.cmu.edu', 'willie@cs.cmu.edu']


import numpy as np


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

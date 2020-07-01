'''
This submodule contains a base class for our models that outlines the methods
that the `BaseAdsorptionDiscoverer` assumes that you have.
'''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'


class BaseModel:
    def __init__(self):
        pass

    def train(self, features, labels):
        '''
        This method needs to train the model to predict the labels given the features
        '''
        raise NotImplementedError

    def predict(self, features):
        '''
        This method needs to predict labels given features
        '''
        raise NotImplementedError

    def save(self):
        '''
        This method needs to somehow save the state of the model so it can be
        loaded later.
        '''
        raise NotImplementedError

    def load(self):
        '''
        This method needs to load the last state of the model
        '''
        raise NotImplementedError

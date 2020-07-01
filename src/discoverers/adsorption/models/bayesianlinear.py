__author__ = 'Willie Neiswanger'
__email__ = 'willie@cs.cmu.edu'


import numpy as np
from sklearn.linear_model import BayesianRidge


class BayesianLinearModel:
    '''
    A Bayesian linear model, implemented via scikit-learn.
    '''

    def __init__(self):
        '''Instantiate sklearn model.'''
        self.model = BayesianRidge()

    def train(self, docs, energies):
        '''Construct a feature representation and fit model.'''
        docs_mat = self.get_docs_mat(docs)
        energies = np.array(energies)
        try:
            self.model.fit(docs_mat, energies)
        except Exception:
            pass

    def predict(self, docs):
        '''For each doc, predict mean and uncertainty.'''

        ### TODO: confirm below is correct when Exception occurs

        try:
            docs_mat = self.get_docs_mat(docs)
            predictions, uncertainties = self.model.predict(docs_mat, return_std=True)
        except Exception:
            predictions = np.zeros(len(docs))
            uncertainties = np.ones(len(docs))
            print('Model prediction: EXCEPTION OCCURED')

        return predictions, uncertainties

    def get_docs_mat(self, docs, fingerprints=True):
        '''Make a random feature representation.'''
        if fingerprints:
            docs_mat = docs
        else:
            docs_list = []
            for d in docs:
                docs_list.append(np.array(d['shift']))
                #
                #rep = d['miller'][:4]
                #rep.append(d['shift'])
                #docs_list.append(np.array(rep))

            docs_mat = np.array(docs_list).reshape(-1, 1)
        return docs_mat

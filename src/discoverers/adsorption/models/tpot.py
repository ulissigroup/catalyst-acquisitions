__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'


import gc
import pickle
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tpot import TPOTRegressor
from gaspy_regress import fingerprinters
from .base import BaseModel


class TPOT(BaseModel):
    '''
    This is our wrapper for fingerprinting sites and then using TPOT to predict
    adsorption energies from those fingerprints.
    '''
    def __init__(self):
        '''
        Instantiate the preprocessing pipeline and the TPOT model
        '''
        # Instantiate the fingerprinter
        inner_fingerprinter = fingerprinters.InnerShellFingerprinter()
        outer_fingerprinter = fingerprinters.OuterShellFingerprinter()
        fingerprinter = fingerprinters.StackedFingerprinter(inner_fingerprinter,
                                                            outer_fingerprinter)
        scaler = StandardScaler()
        pca = PCA()
        preprocessing_pipeline = Pipeline([('fingerprinter', fingerprinter),
                                           ('scaler', scaler),
                                           ('pca', pca)])
        self.preprocessor = preprocessing_pipeline

        # Instantiate TPOT
        self.tpot = TPOTRegressor(generations=2,
                                  population_size=32,
                                  offspring_size=32,
                                  verbosity=2,
                                  scoring='neg_median_absolute_error',
                                  n_jobs=16,
                                  warm_start=True)

    def train(self, docs, energies):
        '''
        Trains both the preprocessor and TPOT in series

        Args:
            docs        List of dictionaries from
                        `gaspy.gasdb.get_adsorption_docs`
            energies    List of floats containing the adsorption energies of
                        `docs`
        '''
        features = self.preprocessor.fit_transform(docs)
        self.tpot.fit(features, energies)

        # Try to address some memory issues by collecting garbage
        _ = gc.collect()  # noqa: F841

    def predict(self, docs):
        '''
        Use the whole fingerprinting and TPOT pipeline to make adsorption
        energy predictions

        Args:
            docs        List of dictionaries from
                        `gaspy.gasdb.get_adsorption_docs`
        Returns:
            predictions     `np.array` of TPOT's predictions of each doc
            uncertainties   `np.array` that contains the "uncertainty
                            prediction" for each site. In this case, it'll
                            just be TPOT's RMSE
        '''
        # Point predictions
        features = self.preprocessor.transform(docs)
        try:
            predictions = np.array(self.tpot.predict(features))
        # In case we need to make a prediction from a loaded state
        except AttributeError:
            predictions = np.array(self.tpot.fitted_pipeline_.predict(features))

        # "Uncertainties" will just be the RMSE
        residuals = np.array([prediction - doc['energy']
                              for prediction, doc in zip(predictions, docs)])
        rmse = np.sqrt((residuals**2).mean())
        uncertainties = np.array([rmse for _ in predictions])

        return predictions, uncertainties

    def save(self):
        '''
        Saves the state of the model into some pickles
        '''
        with open(self._fingerprinter_cache, 'wb') as file_handle:
            pickle.dump(self.model.preprocessor, file_handle)
        with open(self._pipeline_cache, 'wb') as file_handle:
            pickle.dump(self.model.tpot.fitted_pipeline_, file_handle)

    def load(self):
        '''
        Loads a previous state of the model from some pickles
        '''
        with open(self._fingerprinter_cache, 'rb') as file_handle:
            self.model.preprocessor = pickle.load(file_handle)
        with open(self._pipeline_cache, 'rb') as file_handle:
            self.model.tpot.fitted_pipeline_ = pickle.load(file_handle)

    @property
    def _fingerprinter_cache(self):
        return 'fingerprinter.pkl'

    @property
    def _pipeline_cache(self):
        return 'tpot_pipeline.pkl'

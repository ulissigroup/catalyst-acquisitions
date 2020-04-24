'''
This submodule houses the `MultiscaleDiscoverer` child class of
`AdsorptionDiscovererBase` that hallucinates the performance of a new method
that uses a Convolutional-Fed Gaussian Process to perform a sort of multiscale,
hierarchical active discovery method to screen catalysts by screening their
adsorption energies.
'''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'


import os
import pickle
from pathlib import Path
from scipy.stats import norm
from gaspy.gasdb import get_surface_from_doc
from .adsorption_base import AdsorptionDiscovererBase


class MultiscaleDiscoverer(AdsorptionDiscovererBase):
    '''
    This discoverer uses a multi-scale method for selecting new sites with the
    goal of partitioning the search space of bulks into "good" and "not good".

    It does this by performing level set estimation (LSE) for the values of a
    bulk to choose which bulk to study next; then for that bulk it uses active
    learning (AL)/uncertainty sampling to choose which surface in the bulk to
    sample next; then for that surface it uses active optimization (AO) to
    choose which site on the surface to sample next.

    All surrogate model predictions and corresponding uncertainty estimates
    come from a convolution-fed Gaussian process (CFGP).
    '''

    def __init__(self, *args, **kwargs):
        '''
        In addition to the normal things that this class's parent classes do in
        `__init__` this method also instantiates the `CFGPWrapper`
        '''
        self.model = CFGPWrapper()
        self.cache_location = './multiscale_caches/'
        Path(self.cache_location).mkdir(exist_ok=True)
        super().__init__(*args, **kwargs)

    def _train(self, next_batch):
        '''
        Calculate the residuals of the current training batch, then retrain on
        everything

        Arg:
            next_batch  The output of this class's `_choose_next_batch` method
        '''
        # Parse the incoming batch
        try:
            features, dft_energies, next_surfaces = next_batch
        except ValueError:
            features, dft_energies = next_batch
            next_surfaces = [get_surface_from_doc(doc) for doc in features]

        # Calculate and save the results of this next batch
        try:
            tpot_predictions, uncertainties = self.model.predict(features)
            residuals = tpot_predictions - dft_energies
            self.uncertainties.extend(uncertainties)
            self.residuals.extend(residuals.tolist())
        # If prediction doesn't work, then we probably haven't trained the
        # first batch. And if haven't done this, then there's no need to save
        # the residuals and uncertainty estimates.
        except AttributeError:
            pass

        # Retrain
        self.training_features.extend(features)
        self.training_labels.extend(dft_energies)
        self.training_surfaces.extend(next_surfaces)
        self.model.train(self.training_features, self.training_labels)
        self._save_current_run()

    def _choose_next_batch(self):
        '''
        Choose the next batch...?

        Returns:
            features    The features that this method chose to investigate next
            labels      The labels that this method chose to investigate next
            surfaces    The surfaces that this method chose to investigate next
        '''
        # Use the energies to calculate probabilities of selecting each site
        energies, _ = self.model.predict(self.sampling_features)
        gaussian_distribution = norm(loc=self.target_energy, scale=self.assumed_stdev)
        probability_densities = [gaussian_distribution.pdf(energy) for energy in energies]

        # Perform a weighted shuffling of the sampling space such that sites
        # with better energies are more likely to be early in the list
        features, labels, surfaces = self.weighted_shuffle(self.sampling_features,
                                                           self.sampling_labels,
                                                           self.sampling_surfaces,
                                                           weights=probability_densities)
        self.sampling_features = features
        self.sampling_labels = labels
        self.sampling_surfaces = surfaces

        # Now that the samples are sorted, find the next ones and add them to
        # the training set
        features, labels, surfaces = self._pop_next_batch()
        self.training_features.extend(features)
        self.training_labels.extend(labels)
        self.training_surfaces.extend(surfaces)
        return features, labels, surfaces

    def _save_current_run(self):
        '''
        Cache the current point for (manual) warm-starts, because there's a
        solid chance that TPOT might cause a segmentation fault.
        '''
        cache_name = (self.cache_location +
                      '%.3i%s' % (self.next_batch_number, self.cache_affix))
        cache = {key: getattr(self, key) for key in self.cache_keys}
        with open(cache_name, 'wb') as file_handle:
            pickle.dump(cache, file_handle)

    def load_last_run(self):
        '''
        Updates the attributes according to the last cache
        '''
        cache_names = [cache_name for cache_name in os.listdir(self.cache_location)
                       if cache_name.endswith(self.cache_affix)]
        cache_names.sort()
        cache_name = cache_names[-1]
        with open(os.path.join(self.cache_location, cache_name), 'rb') as file_handle:
            cache = pickle.load(file_handle)

        for key, value in cache.items():
            setattr(self, key, value)


class CFGPWrapper:
    '''
    This is our wrapper for using a convolution-fed Gaussian process to predict
    adsorption energies.
    '''
    def __init__(self):
        '''
        Instantiate the convolutional network and the GP
        '''
        raise NotImplementedError

    def train(self, docs, energies):
        '''
        Trains both the network and GP in series

        Args:
            features?   ???
            energies    List of floats containing the adsorption energies
        '''
        raise NotImplementedError

    def predict(self, docs):
        '''
        Use the whole pipeline to make adsorption energy predictions

        Args:
            features?   ???
        Returns:
            predictions     `np.array` of predictions for each site
            uncertainties   `np.array` that contains the "uncertainty
                            prediction" for each site. In this case, it'll
                            be the GP's predicted standard deviation.
        '''
        raise NotImplementedError

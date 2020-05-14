'''
This submodule houses the `MyopicMultiscaleSelection` child class of
`AdsorptionDiscovererBase` that hallucinates the performance of a method that
uses a Convolution-Fed Gaussian Process (CFGP) model to feed predictions into a
myopic, multiscale acquisition function
'''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'


import numpy as np
from scipy.stats import norm
import gpytorch
from torch_geometric.data import DataLoader

from ocpmodels.trainers.simple_trainer import SimpleTrainer
from ocpmodels.models.gps import ExactGP
from ocpmodels.common.lbfgs import FullBatchLBFGS
from ocpmodels.trainers.gpytorch_trainer import GPyTorchTrainer
from ocpmodels.trainers.cfgp_trainer import CfgpTrainer
from ocpmodels.datasets.gasdb import Gasdb

from .adsorption_base import AdsorptionDiscovererBase


class MultiscaleDiscoverer(AdsorptionDiscovererBase):
    '''
    This discoverer uses a multi-scale method for selecting new sites with the
    goal of partitioning the search space of bulks into 'good' and 'not good'.

    It does this by performing level set estimation (LSE) for the values of a
    bulk to choose which bulk to study next; then for that bulk it uses active
    learning (AL)/uncertainty sampling to choose which surface in the bulk to
    sample next; then for that surface it uses active optimization (AO) to
    choose which site on the surface to sample next.

    All surrogate model predictions and corresponding uncertainty estimates
    come from a convolution-fed Gaussian process (CFGP).
    '''
    def __init__(self, db_dir, *args, **kwargs):
        '''
        In addition to the normal things that this class's parent classes do in
        `__init__` this method also instantiates the `CFGPWrapper`

        Args:
            db_dir  String indicating the directory of the `ase.db` dataset we
                    want to search through
            args    See documentation for parent class `AdsorptionDiscovererBase`
            kwargs  See documentation for parent class `AdsorptionDiscovererBase`
        '''
        self.model = CFGPWrapper(db_dir)
        super().__init__(*args, **kwargs)

    @property
    def cache_location(self):
        return './mms_caches/'

    def _train(self, next_batch):
        '''
        Calculate the residuals of the current training batch, then retrain on
        everything

        Arg:
            next_batch  The output of this class's `_choose_next_batch` method
        '''
        indices, dft_energies = next_batch
        next_surfaces = self.get_surfaces_from_indices(indices)

        # Calculate and save the results of this next batch
        try:
            predictions, uncertainties = self.model.predict(indices)
            residuals = predictions - dft_energies
            self.uncertainties.extend(uncertainties)
            self.residuals.extend(residuals.tolist())
        # If prediction doesn't work, then we probably haven't trained the
        # first batch. And if haven't done this, then there's no need to save
        # the residuals and uncertainty estimates.
        except AttributeError:
            pass

        # Retrain
        self.training_features.extend(indices)
        self.training_labels.extend(dft_energies)
        self.training_surfaces.extend(next_surfaces)
        self.model.train(self.training_features)
        self._save_current_run()

    def get_surfaces_from_indices(self, indices):
        surfaces = []

        db = self.model.dataset.ase_db
        for index in indices:
            data = db.get(index).data
            mpid = data['mpid']
            miller = tuple(index for index in data['miller'])
            shift = data['shift']
            top = data['top']
            surface = (mpid, miller, shift, top)
            surfaces.append(surface)

        return surfaces

    def _choose_next_batch(self):
        '''
        Choose the next batch using our Myopic Multiscale Sampling (MMS)
        method.

        Returns:
            features    The indices of the database rows that this method chose
                        to investigate next
            labels      The labels that this method chose to investigate next
        '''
        features = []
        labels = []
        surfaces = []

        # Choose `self.batch_size` samples
        site_energies = self._concatenate_predicted_energies()
        for _ in range(self.batch_size):

            # Use the site energies to pick the bulk/surface/site we want next
            surface_values, bulk_values = self._calculate_surface_and_bulk_values(site_energies)
            mpid = self._select_bulk(bulk_values)
            surface = self._select_surface(mpid, surface_values)
            index, energy = self._select_site(surface, site_energies)

            # Update the batch information
            features.append(index)
            labels.append(energy)
            surfaces.append(surface)

        # Remove the samples from the sampling space
        for feature in features:
            sampling_space_index = self.sampling_features.index(feature)
            del self.sampling_features[sampling_space_index]
            del self.sampling_labels[sampling_space_index]
            del self.sampling_surfaces[sampling_space_index]

        # Add the new batch to the training set
        self.training_features.extend(features)
        self.training_labels.extend(labels)
        self.training_surfaces.extend(surfaces)
        self.next_batch_number += 1
        return features, labels

    def _calculate_surface_and_bulk_values(self, site_energies):
        '''
        Light wrapper to use a set of site energies to calculate surface and
        bulk values.
        '''
        surface_energies = self.calculate_low_coverage_binding_energies_by_surface(site_energies)
        surface_values = self.calculate_surface_values(surface_energies)
        bulk_values = self.calculate_bulk_values(surface_values)
        return surface_values, bulk_values

    def _select_bulk(self, bulk_values):
        '''
        Selects which bulk to sample next using probability of incorrect
        classification. See
        http://www.cs.cmu.edu/~schneide/bryanba_nips2005.pdf for more details.

        Arg:
            bulk_values     The output of the `self.calculate_bulk_values`
                            method.
        Returns:
            mpid    A string indicating the Materials Project ID of the bulk
                    that we should sample next
        '''
        # Figure out the cutoff for the bulk value around which we will be
        # classifying bulks as "good" or "bad"
        all_bulk_values = np.array([values.mean() for values in bulk_values.values()])
        cutoff = norm.ppf(self.quantile_cutoff,
                          loc=all_bulk_values.mean(),
                          scale=all_bulk_values.std())

        # The acquisition value of each bulk is the probability of incorrect
        # classification
        acquisition_values = []
        for mpid, values in bulk_values.items():
            mean = values.mean()
            std = values.std()
            cdf = norm.cdf(x=cutoff, loc=mean, scale=std)
            acq_val = min(cdf, 1-cdf)
            acquisition_values.append((acq_val, mpid))

        # Find the highest acquisition value, then return the corresponding
        # mpid
        acquisition_values.sort(reverse=True)
        mpid = acquisition_values[0][1]
        return mpid

    def _select_surface(self, mpid, surface_values):
        '''
        Selects which surface to sample next using active learning/uncertainty
        sampling. For a better explanation of uncertainty sampling, see
        http://burrsettles.com/pub/settles.activelearning.pdf

        Arg:
            mpid            A string indicating the Materials Project ID of the
                            surface we need to select
            surface_values  The output of the `self.calculate_surface_values`
                            method
        Returns:
            surface     A 4-tuple that contains the (mpid, miller, shift, top)
                        of the surface, where mpid is a string, miller is a
                        3-tuple of integers, the shift is a float, and top is a
                        Boolean
        '''
        acquisition_values = [(values.std(), surface)
                              for surface, values in surface_values.items()
                              if surface[0] == mpid]
        acquisition_values.sort(reverse=True)
        surface = acquisition_values[0][1]
        return surface

    def _select_site(self, surface, site_energies):
        '''
        Selects which site to sample next using active optimization.
        Specifically, we use expected improvement. An explanation of the
        formulas used here can be found on
        http://krasserm.github.io/2018/03/21/bayesian-optimization/.

        Note that this method will also modify one of the standard deviation
        elements in the `site_energies` argument in-place. Specifically:
        When this method "selects" a site, it simultaneously sets the predicted
        uncertainty to 0 to "hallucinate/pretend" that we sampled it. Doing
        this helps us ensure that we won't resample similar calculations within
        a batch of parallel samples.

        Here are the equations we used:
            EI = E[max(f(x) - f(x+), 0)]
               = (mu(x) - f(x+) - xi) * Phi(Z) + sigma(x)*phi(Z) if sigma(x) > 0
                                                                    sigma(x) == 0
            Z = (mu(x) - f(x+) - xi) / sigma(x)     if sigma(x) > 0
                0                                      sigma(x) == 0

        EI = expected improvement
        mu(x) = GP's estimate of the mean value at x
        sigma(x) = GP's standard error/standard deviation estimate at x
        f(x+) = best objective value observed so far
        xi = exploration/exploitation balance factor (higher value promotes exploration)
        Phi(Z) = cumulative distribution function of normal distribution at Z
        phi(Z) = probability distribution function of normal distribution at Z
        Z = test statistic at x

        Arg:
            surface         A 4-tuple that contains the (mpid, miller, shift,
                            top) of the surface, where mpid is a string, miller
                            is a 3-tuple of integers, the shift is a float, and
                            top is a Boolean
            site_energies   The output of the
                            `self._concatenate_predicted_energies` method.
        Returns:
            db_index    An integer indicating the row number of the site within
                        our source database
            dft_energy  The true, DFT-calculated adsorption energy of the
                        selected site
        '''
        # Parse the input
        energies, stdevs, surfaces = site_energies

        # Grab the indices of all the sites
        training_indices = self.training_features
        sampling_indices = self.sampling_features
        indices = training_indices + sampling_indices

        # Parameters for the EI algorithm
        f_best = min(energy for energy, _surface in zip(energies, surfaces)
                     if _surface == surface)
        xi = 0.01

        acquisition_values = []
        for site_index, (db_index, mu, sigma, _surface) in enumerate(zip(indices, energies, stdevs, surfaces)):
            # Only want to consider sites on the correct surface. And if the
            # sigma is zero, then it's a site we've sampled before and
            # therefore do not want to consider again.
            if _surface == surface and sigma:
                imp = mu - f_best - xi
                Z = imp / sigma
                Phi = norm.cdf(Z)
                phi = norm.pdf(Z)
                ei = imp * Phi + sigma * phi
                acquisition_values.append((ei, db_index, site_index))

        # We want the "lowest" EI because we want to find the minimum energy on
        # this surface
        acquisition_values.sort()
        db_index = acquisition_values[0][1]

        # "Hallucinate" the item we just picked by setting its corresponding
        # standard deviation prediction to 0
        site_index = acquisition_values[0][2]
        site_energies[1][site_index] = 0.

        # We also need to get the DFT-calculated energy for later use
        db = self.model.dataset.ase_db
        row = list(db.select(db_index))[0]
        dft_energy = row['data']['adsorption_energy']
        return db_index, dft_energy


class CFGPWrapper:
    '''
    This is our wrapper for using a convolution-fed Gaussian process to predict
    adsorption energies.
    '''
    def __init__(self, db_dir):
        '''
        Instantiate the settings for our CFGP
        '''
        self.__init_conv_trainer(db_dir)
        self.__init_gp_trainer()
        self.trainer = CfgpTrainer(self.cnn_trainer, self.gp_trainer)

    def __init_conv_trainer(self, db_dir):
        task = {'dataset': 'gasdb',
                'description': ('Binding energy regression on a dataset of DFT '
                                'results for CO, H, N, O, and OH adsorption on '
                                'various slabs.'),
                'labels': ['binding energy'],
                'metric': 'mae',
                'type': 'regression'}
        model = {'name': 'cgcnn',
                 'atom_embedding_size': 64,
                 'fc_feat_size': 128,
                 'num_fc_layers': 4,
                 'num_graph_conv_layers': 6}
        dataset = {'src': db_dir}
        optimizer = {'batch_size': 64,
                     'lr_gamma': 0.1,
                     'lr_initial': 0.001,
                     'lr_milestones': [100, 150],
                     'max_epochs': 5,  # per hallucination batch
                     'warmup_epochs': 10,
                     'warmup_factor': 0.2}
        self.cnn_args = {'task': task,
                         'model': model,
                         'dataset': dataset,
                         'optimizer': optimizer,
                         'identifier': 'cnn'}
        self.__set_initial_split()

        self.cnn_trainer = SimpleTrainer(**self.cnn_args)

    def __set_initial_split(self):
        self.dataset = Gasdb(self.cnn_args['dataset'])
        total = self.dataset.ase_db.count()
        train_size = int(0.64 * total)
        val_size = int(0.16 * total)
        test_size = total - train_size - val_size

        self.cnn_args['dataset']['train_size'] = train_size
        self.cnn_args['dataset']['val_size'] = val_size
        self.cnn_args['dataset']['test_size'] = test_size

    def __init_gp_trainer(self):
        self.gp_args = {'Gp': ExactGP,
                        'Optimizer': FullBatchLBFGS,
                        'Likelihood': gpytorch.likelihoods.GaussianLikelihood,
                        'Loss': gpytorch.mlls.ExactMarginalLogLikelihood}

        self.gp_trainer = GPyTorchTrainer(**self.gp_args)

    def train(self, indices):
        '''
        Trains both the network and GP in series

        Args:
            indices     A sequences of integers that map to the row numbers
                        within the database that you want to train on
        '''
        # Make some arbitrary allocation for validation and test datasets. We
        # won't really use them, but pytorch_geometric needs them to be
        # specified to run.
        train_indices = indices
        val_test_indices = list(set(range(len(self.dataset))) - set(train_indices))
        val_test_split_cutoff = int(0.8 * len(val_test_indices))
        val_indices = val_test_indices[:val_test_split_cutoff]
        test_indices = val_test_indices[-val_test_split_cutoff:]

        # Update the data loaders
        train_loader = DataLoader(
            self.dataset[indices],
            batch_size=self.cnn_args['optimizer']['batch_size'],
            shuffle=True,
        )
        val_loader = DataLoader(
            self.dataset[val_indices],
            batch_size=self.cnn_args['optimizer']['batch_size']
        )
        test_loader = DataLoader(
            self.dataset[test_indices],
            batch_size=self.cnn_args['optimizer']['batch_size']
        )
        self.trainer.train_loader = train_loader
        self.trainer.val_loader = val_loader
        self.trainer.test_loader = test_loader
        self.trainer.conv_trainer.train_loader = train_loader
        self.trainer.conv_trainer.val_loader = val_loader
        self.trainer.conv_trainer.test_loader = test_loader

        # Train on the updated data
        self.trainer.train()

    def predict(self, indices):
        '''
        Use the whole pipeline to make adsorption energy predictions

        Arg:
            indices     A sequences of integers that map to the row numbers
                        within the database that you want to train on
        Returns:
            predictions     `np.array` of predictions for each site
            uncertainties   `np.array` that contains the 'uncertainty
                            prediction' for each site. In this case, it'll
                            be the GP's predicted standard deviation.
        '''
        # Decrease all indices by 1 because we're using them to query the
        # `self.dataset` class, which is indexed to 0. But indices are actually
        # indexed to 1, which is consistent with the `self.dataset.ase_db`.
        indices = [index-1 for index in indices]

        data_loader = DataLoader(
            self.dataset[indices],
            batch_size=self.cnn_args['optimizer']['batch_size']
        )

        convs, _ = self.trainer._get_convolutions(data_loader)
        normed_convs = self.trainer.conv_normalizer.norm(convs)
        predictions, uncertainties = self.trainer.gpytorch_trainer.predict(normed_convs)

        predictions = predictions.detach().cpu().numpy()
        uncertainties = uncertainties.detach().cpu().numpy()
        return predictions, uncertainties

'''
This submodule houses the `MyopicMultiscaleSelection` child class of
`AdsorptionDiscovererBase` that hallucinates the performance of a method that
uses a Convolution-Fed Gaussian Process (CFGP) model to feed predictions into a
myopic, multiscale acquisition function
'''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'


import os
import pickle
from pathlib import Path
from scipy.stats import norm
import gpytorch
from torch_geometric.data import DataLoader

# from gaspy.gasdb import get_surface_from_doc
from ocpmodels.trainers.simple_trainer import SimpleTrainer
from ocpmodels.models.gps import ExactGP
from ocpmodels.commow.lbfgs import FullBatchLBFGS
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
        features, dft_energies = next_batch

        # Calculate and save the results of this next batch
        try:
            predictions, uncertainties = self.model.predict(features)
            residuals = predictions - dft_energies
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
        Choose the next batch using our Myopic Multiscale Sampling (MMS)
        method.

        Returns:
            features    The indices of the database rows that this method chose
                        to investigate next
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
        dataset = {'src': db_dir},
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
        self.__set_initial_split(dataset)

        self.cnn_trainer = SimpleTrainer(**self.cnn_args)

    def __set_initial_split(self):
        dataset = Gasdb(self.cnn_args['dataset'])
        total = dataset.ase_db.count()
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

    def train(self, indices, energies):
        '''
        Trains both the network and GP in series

        Args:
            indices     A sequences of integers that map to the row numbers
                        within the database that you want to train on
            energies    List of floats containing the adsorption energies. Not
                        actually needed since the information is implied
                        through the `indices` argument, but stil here to make
                        it consistent with sister classes that need to accept
                        this argument.
        '''
        dataset = Gasdb(**self.cnn_args['dataset'])

        # Make some arbitrary allocation for validation and test datasets. We
        # won't really use them, but pytorch_geometric needs them to be
        # specified to run.
        train_indices = indices
        val_test_indices = list(set(range(len(dataset))) - set(train_indices))
        val_test_split_cutoff = int(0.8 * len(val_test_indices))
        val_indices = val_test_indices[:val_test_split_cutoff]
        test_indices = val_test_indices[-val_test_split_cutoff:]

        train_loader = DataLoader(
            dataset[indices],
            batch_size=self.cnn_args['optimizer']['batch_size'],
            shuffle=True,
        )
        val_loader = DataLoader(
            dataset[val_indices],
            batch_size=self.cnn_args['optimizer']['batch_size']
        )
        test_loader = DataLoader(
            dataset[test_indices],
            batch_size=self.cnn_args['optimizer']['batch_size']
        )

        self.conv_trainer.dataset = dataset
        self.conv_trainer.train_loader = train_loader
        self.conv_trainer.val_loader = val_loader
        self.conv_trainer.test_loader = test_loader

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
        data_loader = DataLoader(
            self.conv_trainer.dataset[indices],
            batch_size=self.cnn_args['optimizer']['batch_size']
        )

        convs, _ = self.trainer._get_convolutions(data_loader)
        normed_convs = self.trainer.conv_normalizer.norm(convs)
        predictions, uncertainties = self.trainer.gpytorch_trainer.predict(normed_convs)

        predictions = predictions.detach().cpu().numpy()
        uncertainties = uncertainties.detach().cpu().numpy()
        return predictions, uncertainties

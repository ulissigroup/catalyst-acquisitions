__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'


import os
import torch
from torch.utils.data import DataLoader
from gpytorch.kernels import MaternKernel
from ocpmodels.common.data_parallel import ParallelCollater
from ocpmodels.trainers.simple_trainer import SimpleTrainer
from ocpmodels.trainers.gpytorch_trainer import GPyTorchTrainer
from ocpmodels.trainers.cfgp_trainer import CfgpTrainer
from ocpmodels.datasets.gasdb import Gasdb
from .base import BaseModel


class CFGP(BaseModel):
    '''
    This is our wrapper for using a convolution-fed Gaussian process to predict
    adsorption energies.
    '''
    def __init__(self, db_dir, num_gpus=torch.cuda.device_count(), CovKernel=None):
        ''' Instantiate the settings for our CFGP '''
        self.db_dir = db_dir
        self.num_gpus = num_gpus
        self.CovKernel = CovKernel
        self.dataset = Gasdb({'src': self.db_dir})

    def train(self, indices, _labels=None):
        '''
        Trains both the network and GP in series

        Args:
            indices     A sequences of integers that map to the row numbers
                        within the database that you want to train on
            _labels     Dummy argument that is not used. It is here to be
                        consistent with the parent class.
        '''
        self._init_cfgp_trainer(indices)

        # Substract 1 from all the indices because they're 1-indexed for the
        # ASE database, but here we'll be using them to grab things from the
        # 0-indexed dataset object.
        train_indices = [index - 1 for index in indices]

        # Update the data loader
        train_loader = DataLoader(
            self.dataset[train_indices],
            batch_size=self.cnn_args['optimizer']['batch_size'],
            shuffle=True,
            collate_fn=self.collate_fn
        )
        self.trainer.train_loader = train_loader
        self.trainer.conv_trainer.train_loader = train_loader

        # Train on the updated data
        self.trainer.train(n_training_iter=100)

        # Clear up some GPU memory
        torch.cuda.empty_cache()

    def _init_cfgp_trainer(self, indices):
        self._init_conv_trainer(self.db_dir, indices)
        self._init_gp_trainer()
        self.trainer = CfgpTrainer(self.cnn_trainer, self.gp_trainer)

    def _init_conv_trainer(self, db_dir, indices):
        task = {'dataset': 'gasdb',
                'description': ('Regression of DFT calculated binding energes'),
                'labels': ['binding energy'],
                'metric': 'mae',
                'type': 'regression'}
        model = {'name': 'cgcnn',
                 'atom_embedding_size': 64,
                 'fc_feat_size': 128,
                 'num_fc_layers': 4,
                 'num_graph_conv_layers': 6,
                 'regress_forces': False}
        dataset = {'src': db_dir,
                   'train_size': len(indices),
                   'val_size': 0,
                   'test_size': 0}
        optimizer = {'batch_size': 64,
                     'lr_gamma': 0.1,
                     'lr_initial': 0.001,
                     'lr_milestones': [25, 45],
                     'max_epochs': 50,  # per hallucination batch
                     'warmup_epochs': 10,
                     'warmup_factor': 0.2,
                     'num_gpus': self.num_gpus}
        self.cnn_args = {'task': task,
                         'model': model,
                         'dataset': dataset,
                         'optimizer': optimizer,
                         'identifier': 'cnn'}

        self.cnn_trainer = SimpleTrainer(**self.cnn_args)
        self.dataset = Gasdb(self.cnn_args['dataset'])

        if self.num_gpus > 1:
            self.collate_fn = ParallelCollater(self.num_gpus)
        else:
            self.collate_fn = None

    def _init_gp_trainer(self):
        self.gp_trainer = GPyTorchTrainer(CovKernel=self.CovKernel)

    def predict(self, indices):
        '''
        Use the whole pipeline to make adsorption energy predictions

        Args:
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
            batch_size=self.cnn_args['optimizer']['batch_size'],
            collate_fn=self.collate_fn
        )

        convs, _ = self.trainer._get_convolutions(data_loader)
        normed_convs = self.trainer.conv_normalizer.norm(convs)
        predictions, uncertainties = self.trainer.gpytorch_trainer.predict(normed_convs)

        predictions = predictions.detach().cpu().numpy()
        uncertainties = uncertainties.detach().cpu().numpy()
        return predictions, uncertainties

    def save(self):
        '''
        Calls the `save_state` method of the `CFGPTrainer` class.
        '''
        self.trainer.save_state()

    def load(self, nn_checkpoint_file=None,
             gp_checkpoint_file='gp_state.pth',
             normalizer_checkpoint_file='normalizer.pth'):
        '''
        Load the `checkpoint.pt` file in the last subfolder within
        `checkpoints/`. Note that once we instantiate this model, we
        automatically create another subfolder. So technically, we look for the
        second-to-last folder to read a `checkpoint.pt` file, which WAS the
        last folder before we made another one.

        Args:
            nn_checkpoint_file          String you can use to override the
                                        checkpoint file to read from. If
                                        `None`, then defaults to what is found
                                        automatically.
            gp_checkpoint_file          String indicating the location of the
                                        GP checkpoint file
            normalizer_checkpoint_file  String indicating the location of the
                                        normalizer checkpoint file
        '''
        if nn_checkpoint_file is None:
            prefix = 'checkpoints'
            cp_folders = os.listdir(prefix)
            cp_folders.sort()
            nn_checkpoint_file = os.path.join(prefix, cp_folders[-2], 'checkpoint.pt')

        self._init_cfgp_trainer([0])
        self.trainer.load_state(nn_checkpoint_file=nn_checkpoint_file,
                                gp_checkpoint_file=gp_checkpoint_file,
                                normalizer_checkpoint_file=normalizer_checkpoint_file)

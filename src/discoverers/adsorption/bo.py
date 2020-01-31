'''
This submodule houses the `BayesianOptimizer` child class of
`AdsorptionDiscovererBase` that hallucinates the performance of Bayesian
Optimization in the context of discovering catalysts by screening their
adsorption energies.
'''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'


import warnings
import pickle
import numpy as np
from scipy.stats import norm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import torch
import gpytorch
from gaspy_regress import fingerprinters
from .benchmarks import AdsorptionDiscovererBase

# The tqdm autonotebook is still experimental, and it warns us. We don't care,
# and would rather not hear about the warning everytime.
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from tqdm.autonotebook import tqdm


class BayesianOptimizer(AdsorptionDiscovererBase):
    '''
    This "discoverer" is actually just a Bayesian optimizer for trying to find
    adsorption energies.
    '''

    def _train(self):
        '''
        Train the GP hyperparameters
        '''
        # Instantiate the preprocessor and GP if we haven't done so already
        if not hasattr(self, 'preprocessor'):
            self._train_preprocessor()

        # Calculate and save the residuals of this next batch
        try:
            ml_energies, _ = self.__make_predictions(self.training_batch)
            dft_energies = np.array([doc['energy'] for doc in self.training_batch])
            residuals = ml_energies - dft_energies
            self.residuals.extend(list(residuals))
        # If this is the very first training batch, then we don't need to save
        # the residuals
        except AttributeError:
            pass

        # Mandatory extension of the training set to include this next batch
        self.training_set.extend(self.training_batch)
        # Re-train on the whole training set
        self.__init_GP()
        _ = self.__train_GP()

    def _train_preprocessor(self):
        '''
        Trains the preprocessing pipeline and assigns it to the `preprocessor`
        attribute.
        '''
        # Open the cached preprocessor
        try:
            cache_name = 'caches/preprocessor.pkl'
            with open(cache_name, 'rb') as file_handle:
                self.preprocessor = pickle.load(file_handle)

        # If there is no cache, then remake it
        except FileNotFoundError:
            inner_fingerprinter = fingerprinters.InnerShellFingerprinter()
            outer_fingerprinter = fingerprinters.OuterShellFingerprinter()
            fingerprinter = fingerprinters.StackedFingerprinter(inner_fingerprinter,
                                                                outer_fingerprinter)
            scaler = StandardScaler()
            pca = PCA()
            preprocessing_pipeline = Pipeline([('fingerprinter', fingerprinter),
                                               ('scaler', scaler),
                                               ('pca', pca)])
            preprocessing_pipeline.fit(self.training_batch)
            self.preprocessor = preprocessing_pipeline

            # Cache it for next time
            with open(cache_name, 'wb') as file_handle:
                pickle.dump(preprocessing_pipeline, file_handle)

    def __init_GP(self):
        '''
        Initialize the exact GP model and assign the appropriate class attributes

        Returns:
            train_x     A `torch.Tensor` of the featurization of the current
                        training set
            train_y     A `torch.Tensor` of the output of the current training set
        '''
        # Grab the initial training data from the current (probably first)
        # training batch
        train_x = torch.Tensor(self.preprocessor.transform(self.training_set))
        train_y = torch.Tensor([doc['energy'] for doc in self.training_set])

        # Initialize the GP
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.GP = ExactGPModel(train_x, train_y, self.likelihood)

        # Optimize the GP hyperparameters
        self.GP.train()
        self.likelihood.train()

        # Set the optimizer that will tune parameters during training:  ADAM
        self.optimizer = torch.optim.Adam([{'params': self.GP.parameters()}], lr=0.1)

        # Set the "loss" function:  marginal log likelihood
        self.loss = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.GP)

        return train_x, train_y

    def __make_predictions(self, docs):
        '''
        Use the GP to make predictions on the current training batch

        Args:
            docs    A list of dictionaries that correspond to the sites you
                    want to make predictions on
        Returns:
            means   A numpy array giving the GP's mean predictions for the
                    `docs` you gave this method.
            stdevs  A numpy array giving the GP's standard deviation/standard
                    error predictions for the `docs` you gave this method.
        '''
        # Get into evaluation (predictive posterior) mode
        self.GP.eval()
        self.likelihood.eval()

        # Make the predictions
        features = torch.Tensor(self.preprocessor.transform(docs))
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions = self.GP(features)

        # Format and return the predictions
        means = predictions.mean.cpu().detach().numpy()
        stdevs = predictions.stddev.cpu().detach().numpy()
        return means, stdevs

    def __train_GP(self):
        '''
        Re-trains the GP on all of the training data
        '''
        # Re-initialize the GP
        train_x, train_y = self.__init_GP()

        # If the loss increases too many times in a row, we will stop the
        # tuning short. Here we initialize some things to keep track of this.
        current_loss = float('inf')
        loss_streak = 0

        # Do at most 50 iterations of training
        for i in tqdm(range(50), desc='GP tuning'):

            # Zero backprop gradients
            self.optimizer.zero_grad()
            # Get output from model
            output = self.GP(train_x)
            # Calc loss and backprop derivatives
            loss = -self.loss(output, train_y)
            loss.backward()
            self.optimizer.step()

            # Stop training if the loss increases twice in a row
            new_loss = loss.item()
            if new_loss > current_loss:
                loss_streak += 1
                if loss_streak >= 2:
                    break
            else:
                current_loss = new_loss
                loss_streak = 0

    def _choose_next_batch(self):
        self.__sort_sampling_space_by_EI()
        self._pop_next_batch()

    def __sort_sampling_space_by_EI(self):
        '''
        Brute-force calculate the expected improvement for each of the sites in
        the sampling space.

        An explanation of the formulas used here can be found on
        http://krasserm.github.io/2018/03/21/bayesian-optimization/

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
        '''
        # Initialize by getting all the GP predictions, the best energy so far,
        # and setting an exploration/exploitation value.
        means, stdevs = self.__make_predictions(self.sampling_space)
        f_best = min(abs(doc['energy'] - self.optimal_value)
                     for doc in self.training_set)
        xi = 0.01

        # Calculate EI for every single point we may sample
        for doc, mu, sigma in zip(self.sampling_space, means, stdevs):

            # Calculate the test statistic
            if sigma > 0:
                Z = (mu - f_best - xi) / sigma
            elif sigma == 0:
                Z = 0.
            else:
                raise RuntimeError('Got a negative standard error from the GP')

            # Calculate EI
            Phi = norm.cdf(Z)
            phi = norm.pdf(Z)
            if sigma > 0:
                EI = (mu - f_best - xi)*Phi + sigma*phi
            elif sigma == 0:
                EI = 0.

            # Save the EI results directly to the sampling space, then sort our
            # sampling space by it. High values of EI will show up first in the
            # list.
            doc['EI'] = EI
        self.sampling_space.sort(key=lambda doc: doc['EI'], reverse=True)


class ExactGPModel(gpytorch.models.ExactGP):
    '''
    We will use the simplest form of GP model with exact inference. This is
    taken from one of GPyTorch's tutorials.
    '''
    def __init__(self, train_x, train_y, likelihood):
        '''
        Args:
            train_x     A numpy array with your training features
            train_y     A numpy array with your training labels
            likelihood  An instance of one of the `gpytorch.likelihoods.*` classes
        '''
        # Convert the training data into tensors, which GPyTorch needs to run
        train_x = torch.Tensor(train_x)
        train_y = torch.Tensor(train_y)

        # Initialize the model
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

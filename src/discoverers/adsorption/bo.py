'''
This submodule houses the `BayesianOptimizer` child class of
`AdsorptionDiscovererBase` that hallucinates the performance of Bayesian
Optimization in the context of discovering catalysts by screening their
adsorption energies.
'''

__author__ = 'Kevin Tran and Willie Neiswanger'
__email__ = 'ktran@andrew.cmu.edu and willie@cs.cmu.edu'


import warnings
import random
import numpy as np
from scipy.stats import norm
from sklearn.linear_model import BayesianRidge
from .adsorption_base import AdsorptionDiscovererBase

# The tqdm autonotebook is still experimental, and it warns us. We don't mind,
# and would rather not hear about the warning everytime.
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from tqdm.autonotebook import tqdm


class BayesianOptimizer(AdsorptionDiscovererBase):
    '''
    This "discoverer" is actually just a Bayesian optimizer for trying to find
    adsorption energies.
    '''

    def __init__(self, *args, **kwargs):
        '''
        Instantiate self.model.
        '''
        self.model = BayesianLinearModel()
        super().__init__(*args, **kwargs)

    def _train(self, next_batch):
        '''
        Calculate the residuals of the current training batch, then retrain on
        everything.

        Args:
            next_batch  The output of this class's `_choose_next_batch` method
        '''
        # Train at the start ### TODO: remove this?
        self.model.train(self.training_features, self.training_labels)

        # Parse the incoming batch
        try:
            features, dft_energies, next_surfaces = next_batch
        except ValueError:
            features, dft_energies = next_batch
            next_surfaces = [_ for _ in features]
            #next_surfaces = [get_surface_from_doc(doc) for doc in features]

        # Get predictions and uncertainties from self.model
        predictions, uncertainties = self.model.predict(features)
        residuals = predictions - dft_energies
        self.uncertainties.extend(uncertainties)
        self.residuals.extend(residuals.tolist())

        # Retrain
        self.training_features.extend(features)
        self.training_labels.extend(dft_energies)
        self.training_surfaces.extend(next_surfaces)
        self.model.train(self.training_features, self.training_labels)

    def _choose_next_batch(self):
        self.__sort_sampling_space_by_EI()
        features, labels, surfaces = self._pop_next_batch()
        return features, labels, surfaces

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

        means, stds = self.model.predict(self.sampling_features)
        f_best = min(abs(energy - self.target_energy)
                     for energy in self.training_labels)
        xi = 0.01

        # Compute EI for every point we might choose 
        ei_list = []
        for energy, mu, sigma in zip(self.sampling_labels, means, stds):

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

            ei_list.append(EI)

        # Zip sampling features, labels, and ei values; sort by ei; then unpack
        sampling_tup = list(zip(self.sampling_features, self.sampling_labels, ei_list))
        sampling_tup.sort(key=lambda tup: tup[2], reverse=True)
        self.sampling_features = [tup[0] for tup in sampling_tup]
        self.sampling_labels = [tup[1] for tup in sampling_tup]


class BayesianLinearModel:
    '''
    A Bayesian linear model, implemented via scikit-learn.
    '''
    
    def __init__(self):
        '''Instantiate sklearn model.'''
        self.model = BayesianRidge()

    def train(self, docs, energies):
        '''Construct a feature representation and fit model.'''
        docs_vector = self.get_docs_vector(docs)
        energies = np.array(energies)
        try:
            self.model.fit(docs_vector, energies)
        except Exception:
            pass

    def predict(self, docs):
        '''For each doc, predict mean and uncertainty.'''

        ### TODO: confirm below is correct when Exception occurs

        try:
            docs_vector = self.get_docs_vector(docs)
            predictions, uncertainties = self.model.predict(docs_vector, return_std=True)
        except Exception:
            predictions = np.zeros(len(docs))
            uncertainties = np.ones(len(docs))

        return predictions, uncertainties

    def get_docs_vector(self, docs):
        '''Make a random feature representation.'''
        docs_vector = []
        for d in docs:
            docs_vector.append(np.array(d['shift']))
            #
            #rep = d['miller'][:4]
            #rep.append(d['shift'])
            #docs_vector.append(np.array(rep))

        docs_vector = np.array(docs_vector).reshape(-1, 1)
        return docs_vector



#class ExactGPModel(gpytorch.models.ExactGP):
    #'''
    #We will use the simplest form of GP model with exact inference. This is
    #taken from one of GPyTorch's tutorials.
    #'''
    #def __init__(self, train_x, train_y, likelihood):
        #'''
        #Args:
            #train_x     A numpy array with your training features
            #train_y     A numpy array with your training labels
            #likelihood  An instance of one of the `gpytorch.likelihoods.*` classes
        #'''
        ## Convert the training data into tensors, which GPyTorch needs to run
        #train_x = torch.Tensor(train_x)
        #train_y = torch.Tensor(train_y)

        ## Initialize the model
        #super().__init__(train_x, train_y, likelihood)
        #self.mean_module = gpytorch.means.ConstantMean()
        #self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    #def forward(self, x):
        #mean_x = self.mean_module(x)
        #covar_x = self.covar_module(x)
        #return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    #def _train_preprocessor(self):
        #'''
        #Trains the preprocessing pipeline and assigns it to the `preprocessor`
        #attribute.
        #'''
        ## Open the cached preprocessor
        #try:
            #cache_name = 'caches/preprocessor.pkl'
            #with open(cache_name, 'rb') as file_handle:
                #self.preprocessor = pickle.load(file_handle)

        ## If there is no cache, then remake it
        #except FileNotFoundError:
            #inner_fingerprinter = fingerprinters.InnerShellFingerprinter()
            #outer_fingerprinter = fingerprinters.OuterShellFingerprinter()
            #fingerprinter = fingerprinters.StackedFingerprinter(inner_fingerprinter,
                                                                #outer_fingerprinter)
            #scaler = StandardScaler()
            #pca = PCA()
            #preprocessing_pipeline = Pipeline([('fingerprinter', fingerprinter),
                                               #('scaler', scaler),
                                               #('pca', pca)])
            #preprocessing_pipeline.fit(self.training_batch)
            #self.preprocessor = preprocessing_pipeline

            ## Cache it for next time
            #with open(cache_name, 'wb') as file_handle:
                #pickle.dump(preprocessing_pipeline, file_handle)

    #def __init_GP(self):
        #'''
        #Initialize the exact GP model and assign the appropriate class attributes

        #Returns:
            #train_x     A `torch.Tensor` of the featurization of the current
                        #training set
            #train_y     A `torch.Tensor` of the output of the current training set
        #'''
        ## Grab the initial training data from the current (probably first)
        ## training batch
        #train_x = torch.Tensor(self.preprocessor.transform(self.training_set))
        #train_y = torch.Tensor([doc['energy'] for doc in self.training_set])

        ## Initialize the GP
        #self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        #self.GP = ExactGPModel(train_x, train_y, self.likelihood)

        ## Optimize the GP hyperparameters
        #self.GP.train()
        #self.likelihood.train()

        ## Set the optimizer that will tune parameters during training:  ADAM
        #self.optimizer = torch.optim.Adam([{'params': self.GP.parameters()}], lr=0.1)

        ## Set the "loss" function:  marginal log likelihood
        #self.loss = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.GP)

        #return train_x, train_y

    #def __make_predictions(self, docs):
        #'''
        #Use the GP to make predictions on the current training batch

        #Args:
            #docs    A list of dictionaries that correspond to the sites you
                    #want to make predictions on
        #Returns:
            #means   A numpy array giving the GP's mean predictions for the
                    #`docs` you gave this method.
            #stdevs  A numpy array giving the GP's standard deviation/standard
                    #error predictions for the `docs` you gave this method.
        #'''
        ## Get into evaluation (predictive posterior) mode
        #self.GP.eval()
        #self.likelihood.eval()

        ## Make the predictions
        #features = torch.Tensor(self.preprocessor.transform(docs))
        #with torch.no_grad(), gpytorch.settings.fast_pred_var():
            #predictions = self.GP(features)

        ## Format and return the predictions
        #means = predictions.mean.cpu().detach().numpy()
        #stdevs = predictions.stddev.cpu().detach().numpy()
        #return means, stdevs

    #def __train_GP(self):
        #'''
        #Re-trains the GP on all of the training data
        #'''
        ## Re-initialize the GP
        #train_x, train_y = self.__init_GP()

        ## If the loss increases too many times in a row, we will stop the
        ## tuning short. Here we initialize some things to keep track of this.
        #current_loss = float('inf')
        #loss_streak = 0

        ## Do at most 50 iterations of training
        #for i in tqdm(range(50), desc='GP tuning'):

            ## Zero backprop gradients
            #self.optimizer.zero_grad()
            ## Get output from model
            #output = self.GP(train_x)
            ## Calc loss and backprop derivatives
            #loss = -self.loss(output, train_y)
            #loss.backward()
            #self.optimizer.step()

            ## Stop training if the loss increases twice in a row
            #new_loss = loss.item()
            #if new_loss > current_loss:
                #loss_streak += 1
                #if loss_streak >= 2:
                    #break
            #else:
                #current_loss = new_loss
                #loss_streak = 0

'''
This submodule houses the `BayesianOptimizer` child class of
`BaseAdsorptionDiscoverer` that hallucinates the performance of Bayesian
Optimization in the context of discovering catalysts by screening their
adsorption energies.
'''

__author__ = 'Willie Neiswanger'
__email__ = 'willie@cs.cmu.edu'


from scipy.stats import norm
from .base import BaseAdsorptionDiscoverer


class BayesianOptimizer(BaseAdsorptionDiscoverer):
    '''
    This "discoverer" is actually just a Bayesian optimizer for trying to find
    adsorption energies.
    '''

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

'''
This submodule houses the `MyopicMultiscaleSelection` child class of
`AdsorptionDiscovererBase` that hallucinates the performance of a method that
uses a Convolution-Fed Gaussian Process (CFGP) model to feed predictions into a
myopic, multiscale acquisition function
'''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'


from collections import defaultdict
import numpy as np
from scipy.stats import norm
from .base import BaseAdsorptionDiscoverer


class MultiscaleDiscoverer(BaseAdsorptionDiscoverer):
    '''
    This discoverer uses a multi-scale method for selecting new sites with the
    goal of partitioning the search space of bulks into 'good' and 'not good'.

    It does this by performing level set estimation (LSE) for the values of a
    bulk to choose which bulk to study next; then for that bulk it uses active
    learning (AL)/uncertainty sampling to choose which surface in the bulk to
    sample next; then for that surface it uses active optimization (AO) to
    choose which site on the surface to sample next.
    '''

    def _choose_next_batch(self):
        '''
        Choose the next batch using our Myopic Multiscale Sampling (MMS)
        method.

        Returns:
            features    The list of indices of the database rows that this
                        method chose to investigate next
            labels      The list of labels that this method chose to
                        investigate next
            surfaces    The list of surfaces that this method chose to
                        investigate next
        '''
        features = []
        labels = []
        surfaces = []

        # Choose `self.batch_size` samples
        site_energies = self._concatenate_predicted_energies()
        for _ in range(self.batch_size):

            # Use the site energies to pick the bulk/surface/site we want next
            surface_values, bulk_values = self._calculate_surface_and_bulk_values(site_energies)
            ordered_bulks = self._prioritize_bulks(bulk_values)
            ordered_surfaces = self._prioritize_surfaces(surface_values)
            index, energy, surface = self._select_site(ordered_bulks, ordered_surfaces, site_energies)

            # Remove the samples from the sampling space
            try:
                sampling_space_index = self.sampling_features.index(index)
                assert index == self.sampling_features[sampling_space_index]
                assert energy == self.sampling_labels[sampling_space_index]
                assert surface == self.sampling_surfaces[sampling_space_index]
                del self.sampling_features[sampling_space_index]
                del self.sampling_labels[sampling_space_index]
                del self.sampling_surfaces[sampling_space_index]

                # Update the batch information
                features.append(index)
                labels.append(energy)
                surfaces.append(surface)

            # If the index is not in the sampling space, then we're probably
            # done the hallucination.
            except ValueError:
                assert len(self.sampling_features) == 0
                break

        self.next_batch_number += 1
        return features, labels, surfaces

    def _calculate_surface_and_bulk_values(self, site_energies):
        '''
        Light wrapper to use a set of site energies to calculate surface and
        bulk values.
        '''
        surface_energies = self.calculate_low_coverage_binding_energies_by_surface(site_energies)
        surface_values = self.calculate_surface_values(surface_energies)
        bulk_values = self.calculate_bulk_values(surface_values)
        return surface_values, bulk_values

    def _prioritize_bulks(self, bulk_values):
        '''
        Selects which bulk to sample next using probability of incorrect
        classification. See
        http://www.cs.cmu.edu/~schneide/bryanba_nips2005.pdf for more details.

        Args:
            bulk_values     The output of the `self.calculate_bulk_values`
                            method.
        Returns:
            ordered_bulks   An ordered list of the bulk Materials Project IDs
                            where things earlier in the list should be sampled
                            first.
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
            if not np.isnan(acq_val):
                acquisition_values.append((acq_val, mpid))
            # If the acquistion value calculation "falied", give it a value of
            # 0 so we don't prioritize it highly
            else:
                acquisition_values.append((0., mpid))

        # Order the bulks
        acquisition_values.sort(reverse=True)
        ordered_bulks = [mpid for _, mpid in acquisition_values]
        return ordered_bulks

    def _prioritize_surfaces(self, surface_values):
        '''
        Selects which surface to sample next using active learning/uncertainty
        sampling. For a better explanation of uncertainty sampling, see
        http://burrsettles.com/pub/settles.activelearning.pdf

        Args:
            surface_values  The output of the `self.calculate_surface_values`
                            method
        Returns:
            ordered_surfaces    A dictionary whose keys are the Materials
                                Project IDs of all the bulks and whose values
                                are ordered lists of the surfaces for that bulk
                                where things earlier in the list should be
                                sampled first.
        '''
        acquisition_values = [(np.std(values), surface)
                              for surface, values in surface_values.items()]
        acquisition_values.sort(reverse=True)

        ordered_surfaces = defaultdict(list)
        for _, surface in acquisition_values:
            mpid = surface[0]
            ordered_surfaces[mpid].append(surface)
        return ordered_surfaces

    def _select_site(self, ordered_bulks, ordered_surfaces, site_energies):
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

        Args:
            ordered_bulks       The output of the `self._prioritize_bulks` method.
            ordered_surfaces    The output of the `self._prioritize_surfaces` method.
            site_energies       The output of the
                                `self._concatenate_predicted_energies` method.
        Returns:
            db_index    An integer indicating the row number of the site within
                        our source database
            dft_energy  The true, DFT-calculated adsorption energy of the
                        selected site
            surface     A 4-tuple that contains the information needed to
                        define the surface that the site sits on. Should
                        contain (mpid, miller, shift, top).
        '''
        # Tuning parameter for EI
        xi = 0.01

        # Parse/package the input for faster processing
        features, energies, stdevs, surfaces = site_energies
        parsed_sites = {mpid: {surface: [] for surface in ordered_surfaces[mpid]}
                        for mpid in ordered_bulks}
        for site_index, (db_index, energy, std, surface) in enumerate(zip(features, energies, stdevs, surfaces)):
            mpid = surface[0]
            # Only want to consider sites on the correct surface. And if the
            # sigma is zero, then it's a site we've sampled before and
            # therefore do not want to consider again.
            if std > 0 and not np.isnan(std):
                parsed_sites[mpid][surface].append((site_index, db_index, energy, std))

        # Loop through all the MPIDs/surfaces in case the first few of each
        # don't have any valid sites to sample
        site_found = False
        for mpid in ordered_bulks:
            for surface in ordered_surfaces[mpid]:
                try:
                    f_best = min(energy for _, _, energy, _ in parsed_sites[mpid][surface])
                except ValueError:  # If there are no sites to sample, move to the next surface
                    continue

                # Calculate the EI for each site
                acquisition_values = []
                for site_index, db_index, mu, sigma in parsed_sites[mpid][surface]:
                    imp = mu - f_best - xi
                    Z = imp / sigma
                    Phi = norm.cdf(Z)
                    phi = norm.pdf(Z)
                    ei = imp * Phi + sigma * phi
                    acquisition_values.append((ei, db_index, site_index))

                # We want the "lowest" EI because we want to find the minimum
                # energy on this surface
                acquisition_values.sort()
                if len(acquisition_values) > 0:
                    site_found = True
                    _, db_index, site_index = acquisition_values[0]

                    # Exit the search if we've found a site successfully
                    break
            if site_found is True:
                break
        assert site_found is True

        # "Hallucinate" the item we just picked by setting its corresponding
        # standard deviation prediction to 0
        site_energies[2][site_index] = 0.

        # We also need to get the DFT-calculated energy for later use
        db = self.model.dataset.ase_db
        row = list(db.select(db_index))[0]
        dft_energy = row['data']['adsorption_energy']
        return db_index, dft_energy, surface

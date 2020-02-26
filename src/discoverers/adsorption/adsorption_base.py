'''
This submodule is meant to establish performance benchmarks for catalyst
discovery via screening of adsorption energies.
'''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'


from copy import deepcopy
import numpy as np
from scipy.stats import norm
from matplotlib import ticker
from ..base import ActiveDiscovererBase


# Used to put commas into figures' axes' labels
FORMATTER = ticker.FuncFormatter(lambda x, p: format(int(x), ','))


class AdsorptionDiscovererBase(ActiveDiscovererBase):
    '''
    Here we extend the `ActiveDiscovererBase` class while making the following
    assumptions:  1) we are trying to optimize the adsorption energy and 2) our
    inputs are a list of dictionaries with the 'energy' and 'std' keys.
    '''
    def __init__(self, target_energy, quantile_cutoff,
                 training_features, training_labels, training_surfaces,
                 sampling_features, sampling_labels, sampling_surfaces,
                 n_samples=20, alpha=1., beta=1.,
                 batch_size=200, init_train=True):
        '''
        Perform the initial training for the active discoverer, and then
        initialize all of the other attributes.

        Args:
            target_energy       The optimal adsorption energy of an adsorption
                                site.
            quantile_cutoff     A float within (0, 1). When we search for bulk
                                materials, we want to classify them as "good"
                                or "not good". "Good" is defined as whether a
                                bulk is in the top quantile of the total set of
                                bulks we are searching over. This
                                `quantile_cutoff` argument is the threshold at
                                which we consider a bulk good or not. For
                                example:  A value of 0.95 will search for the
                                top 5% of bulks; a value of 0.80 will search
                                for the top 20% of bulks; etc.
            training_features   A sequence that contains the features that can
                                be used to train/initialize the surrogate model
                                of the active discoverer.
            training_labels     A sequence that contains the labels that can
                                be used to train/initialize the surrogate model
                                of the active discoverer.
            training_surfaces   A sequence that contains 4-tuples that represent
                                the surfaces of each site in the training set.
                                The 4-tuple should contain (mpid, miller,
                                shift, top).
            sampling_features   A sequence containing the features for the rest
                                of the possible sampling space.
            sampling_labels     A sequence containing the labels for the rest
                                of the possible sampling space.
            sampling_surfaces   A sequence that contains 4-tuples that represent
                                the surfaces of each site in the sampling
                                space. The 4-tuple should contain (mpid,
                                miller, shift, top).
            n_samples           An integer indicating how many times we want to
                                sample all the distributions of values. We do
                                this sampling so that we can propagate
                                uncertainty from the site-level to the
                                bulk-level.
            alpha               A float for the pre-exponential factor of the
                                Arrhenius relationship between energy and value.
            beta                A float for the exponential factor of the
                                Arrhenius relationship between energy and
                                value; akin to the activation energy.
            batch_size          An integer indicating how many elements in the
                                sampling space you want to choose during each
                                batch of discovery. Defaults to 200.
            init_train          Boolean indicating whether or not you want to do
                                your first training during this initialization.
                                You should keep it `True` unless you're doing a
                                warm start (manually).
        '''
        super().__init__(training_features, training_labels,
                         sampling_features, sampling_labels,
                         batch_size=batch_size, init_train=init_train)

        self.target_energy = target_energy
        self.n_samples = n_samples
        self.alpha = alpha
        self.beta = beta

        if not (0 < quantile_cutoff < 1):
            raise ValueError('The quantile cutoff should be between 0 and 1, '
                             'but is actually %.3f.' % quantile_cutoff)
        else:
            self.quantile_cutoff = quantile_cutoff

    def _update_reward(self):
        '''
        For catalyst discovery via adsorption energy screening, we use a
        hierarchical method for calculating the expected reward/value of our
        discovery framework.

        This method involves finding the low-coverage binding energy of each
        catalyst surface we are considering, and then combining that
        low-coverage binding energy with Sabatier scaling relationships to
        calculate the expected activity of that surface. Then we average the
        expected activity of all surfaces within a bulk to determine the
        expected activity of a bulk. Once we have the expected activity of all
        bulks (or to be more precise, the probability distributions of the
        activities of each bulk), then we judge our current accuracy in
        predicting whether or not the bulk will be active. The accuracy of our
        judgement (across the entire search space is our reward. Accuracy is
        quantified via F1 score for the binary classification of "is it active
        or not".

        This method will calculate our reward so far.

        Returns:
            f_one   We use our model's predictions of adsorption energies to
                    calculate the values of bulks. Then we use our estimates of
                    the bulk values to do a binary classification:  Is a bulk
                    at the top X% of the distribution of all bulks, where X is
                    `1 - self.quantile_cutoff`? The `f_one` float that we
                    return is the F1 score of this binary classifier.
        '''
        # Calculate current bulk classes
        current_bulk_values = self.calculate_bulk_values(current=True)
        current_bulk_classes = self._classify_bulks(current_bulk_values)

        # Use the two different sets of bulk values to calculate the F1 scores,
        # then set the F1 score as the reward
        precision = self._calculate_precision(current_bulk_classes)
        recall = self._calculate_recall(current_bulk_classes)
        f_one = 2 * (precision * recall) / (precision + recall)
        return f_one

    def calculate_bulk_values(self, current=True):
        '''
        Calculates the distributions of values of each bulk. Requires the
        `self.model` attribute to be able to accept `self.sampling_features`
        and return a 2-tuple of sequences:  the first sequence should be the
        predicted labels and the second sequence should be the predicted
        standard deviations/uncertainties.

        Args:
            current     A Boolean indicating whether you want the "current"
                        results or the final ones. "Current" results come from
                        data aggregations of already "sampled" points with zero
                        uncertainty and "unsampled" points that whose means and
                        uncertainties are calculated by `self.model`. So if
                        this argument is `True`, then this will return what the
                        hallucination should think the current state should be.
                        If `False`, then this will return the true results as
                        per all the real data fed into it.
        Returns:
            values_by_surface   A dictionary whose keys are the bulk identifier
                                and whose values are a `np.array` of floats
                                indicating the "value" of each bulk.
        '''
        values_by_surface = self.calculate_surface_values(current)

        # Concatenate all the values for each surface onto their corresponding
        # bulks
        surface_values_by_bulk = {}
        for surface, values in values_by_surface.items():
            bulk_id = surface[0]
            surface_values = values.reshape((1, -1))
            try:
                surface_values_by_bulk[bulk_id] = np.concatenate((surface_values_by_bulk[bulk_id], surface_values), axis=0)
            except KeyError:
                surface_values_by_bulk[bulk_id] = surface_values

        # The value of a bulk is the average value of all of its surfaces
        bulk_values = {bulk_id: surface_values.mean(axis=0)
                       for bulk_id, surface_values in surface_values_by_bulk.items()}
        return bulk_values

    def calculate_surface_values(self, current=True):
        '''
        Calculates the "value" of each surface in the discovery space by
        assuming an Arrhenius-like relationship between the low coverage
        binding energy of the surface and its value.

        Args:
            current     A Boolean indicating whether you want the "current"
                        results or the final ones. "Current" results come from
                        data aggregations of already "sampled" points with zero
                        uncertainty and "unsampled" points that whose means and
                        uncertainties are calculated by `self.model`. So if
                        this argument is `True`, then this will return what the
                        hallucination should think the current state should be.
                        If `False`, then this will return the true results as
                        per all the real data fed into it.
        Returns:
            values_by_surface   A dictionary whose keys are a 4-tuple
                                containing surface information (mpid, miller,
                                shift, top) and whose values are a `np.array`
                                of floats indicating the "value" of a surface.
        '''
        energies_by_surface = self.calculate_low_coverage_binding_energies_by_surface(current)

        # Perform an Arrhenius-like transformation of the binding energies to
        # get a rough estimate of value/activity.
        values_by_surface = {}
        for surface, energies in energies_by_surface.items():
            energy_diffs = np.abs(energies - self.target_energy)
            value = self.alpha * np.exp(-self.beta * energy_diffs)
            values_by_surface[surface] = value

        return values_by_surface

    def calculate_low_coverage_binding_energies_by_surface(self, current=True):
        '''
        Find/predicts the low coverage binding energies for each surface in the
        discovery space. Uses both DFT data (with zero uncertainty) and ML data
        (with predicted uncertainty).

        Arg:
            current     A Boolean indicating whether you want the "current"
                        results or the final ones. "Current" results come from
                        data aggregations of already "sampled" points with zero
                        uncertainty and "unsampled" points that whose means and
                        uncertainties are calculated by `self.model`. So if
                        this argument is `True`, then this will return what the
                        hallucination should think the current state should be.
                        If `False`, then this will return the true results as
                        per all the real data fed into it.
        Returns:
            low_cov_energies_by_surface     A dictionary whose keys are a
                                            4-tuple containing surface
                                            information (mpid, miller, shift,
                                            top) and whose values are a
                                            `np.array` of floats indicating the
                                            sampled low coverage adsorption
                                            energies of each surface.
        '''
        # Grab the correct dataset
        if current is True:
            energies, stdevs, surfaces = self._concatenate_predicted_energies()
        elif current is False:
            energies, stdevs, surfaces = self._concatenate_true_energies()
        else:
            raise ValueError('The "current" argument should be Boolean, but is '
                             'instead %s.' % type(current))

        # "Sample" all the sites `self.n_samples` times
        energies_by_surface = {}
        for energy, stdev, surface in zip(energies, stdevs, surfaces):
            samples = norm.rvs(loc=energy, scale=stdev, size=self.n_samples)
            samples = np.array(samples).reshape((1, -1))
            try:
                energies_by_surface[surface] = np.concatenate((energies_by_surface[surface], samples), axis=0)
            except KeyError:
                energies_by_surface[surface] = samples

        # Grab the lowest energy from each surface for each of the samples
        low_cov_energies_by_surface = {surface: sampled_energies.min(axis=0)
                                       for surface, sampled_energies in energies_by_surface.items()}
        return low_cov_energies_by_surface

    def _concatenate_predicted_energies(self):
        '''
        This method will return the adsorption energies and corresponding
        uncertainty estimates on the entire discovery space. If something has
        already been sampled, the energy will come from DFT and uncertainty
        will be 0. If something has not been sampled, the energy and its
        uncertainty will come from the surrogate model.

        Returns:
            energies    A sequence containing all the energies of everything in
                        the discovery space (including both sampled and
                        unsampled sites).
            stdevs      A sequence containing corresponding uncertainty estimates
                        for the `energies` object. Will be set to 0 for sampled
                        sites. Otherwise will be calculated by `self.model`.
            surfaces    A list of 4-tuples that contain the information needed
                        to figure out which surface this site sits on. Should
                        contain (mpid, miller, shift, top).
        '''
        # Get the energies of things we've already "sampled". We also set their
        # uncertainties to 0 because we "know" what their values are.
        sampled_energies = deepcopy(self.training_labels)
        sampled_stdevs = [0. for _ in sampled_energies]
        sampled_surfaces = deepcopy(self.training_surfaces)

        # Use the model to make predictions on the unsampled space
        unsampled_features = deepcopy(self.sampling_features)
        predicted_energies, predicted_stdevs = self.model.predict(unsampled_features)
        unsampled_surfaces = deepcopy(self.sampling_surfaces)

        # Put it all together
        energies = sampled_energies + predicted_energies
        stdevs = sampled_stdevs + predicted_stdevs
        surfaces = sampled_surfaces + unsampled_surfaces
        return energies, stdevs, surfaces

    def _concatenate_true_energies(self):
        '''
        This method will return the adsorption energies and corresponding
        "uncertainty estimates" on the entire discovery space. It will get data
        only from DFT results and always return an uncertainty of 0.

        Returns:
            energies    A sequence containing all the energies of everything in
                        the discovery space (including both sampled and
                        unsampled sites).
            stdevs      A sequence containing corresponding uncertainty estimates
                        for the `energies` object. Will be set to 0 for
                        everything.
            surfaces    A list of 4-tuples that contain the information needed
                        to figure out which surface this site sits on. Should
                        contain (mpid, miller, shift, top).
        '''
        # Get the energies of things we've already "sampled". We also set their
        # uncertainties to 0 because we "know" what their values are.
        sampled_energies = deepcopy(self.training_labels)
        sampled_stdevs = [0. for _ in sampled_energies]
        sampled_surfaces = deepcopy(self.training_surfaces)

        # Grab the results from the unsampled sites
        unsampled_energies = deepcopy(self.sampling_labels)
        unsampled_stdevs = [0. for _ in unsampled_energies]
        unsampled_surfaces = deepcopy(self.sampling_surfaces)

        # Put it all together
        energies = sampled_energies + unsampled_energies
        stdevs = sampled_stdevs + unsampled_stdevs
        surfaces = sampled_surfaces + unsampled_surfaces
        return energies, stdevs, surfaces

    def _classify_bulks(self, bulk_values):
        '''
        Uses the true bulk values to classify each bulk as "good" or "not good"
        according to whether or not its bulk value quantile is above or below
        the `self.quantile_cutoff', respectively.

        Arg:
            bulk_values     A dictionary whose keys are the bulk ids and whose
                            values are... the value of the bulk. Yeah this
                            naming convention isn't the best. See
                            `self.calculate_bulk_values`.
        Returns:
            good_bulks  A dictionary whose values are the bulk ids and whose
                        values are Booleans. `True` means that the bulk is
                        above the threshold, and `False` means that it is
                        below.
        '''
        # Sort all the bulks by their median value. Higher valued bulks will
        # show up first.
        sorted_bulks = [(bulk, np.median(values))
                        for bulk, values in bulk_values.items()]
        sorted_bulks.sort(key=lambda tuple_: tuple_[1], reverse=True)

        # Classify the bulks as good (`True`) if they are within the quantile
        # threshold. Otherwise, consider them "bad".
        cutoff = round((1-self.quantile_cutoff) * len(sorted_bulks))
        good_bulks = {bulk: True if i <= cutoff else False
                      for i, (bulk, value) in sorted_bulks}
        return good_bulks

    def _calculate_precision(self, current_bulk_classes):
        '''
        Calculates the precision of our binary classifier (see
        `self._update_reward`).

        Arg:
            current_bulk_classes    The output of `self._classify_bulks` when
                                    you give in the current bulk values
        Returns:
            recall  The precision of our binary classifier (see
                    `self._update_reward`)
        '''
        # Initialize
        final_bulk_classes = self._calculate_final_bulk_classes()
        true_positives = 0
        false_positives = 0

        # Count the number of true positives and false positives
        for bulk, final_class in final_bulk_classes.items():
            current_class = current_bulk_classes[bulk]
            if current_class is True:
                if final_class is True:
                    true_positives += 1
                else:
                    false_positives += 1

        # Calculate precision
        precision = true_positives / (true_positives + false_positives)
        return precision

    def _calculate_recall(self, current_bulk_classes):
        '''
        Calculates the recall of our binary classifier (see
        `self._update_reward`).

        Arg:
            current_bulk_classes    The output of `self._classify_bulks` when
                                    you give in the current bulk values
        Returns:
            recall  The recall of our binary classifier (see
                    `self._update_reward`)
        '''
        # Initialize
        final_bulk_classes = self._calculate_final_bulk_classes()
        true_positives = 0
        actual_positives = 0

        # Count the number of actual positives and the number of correctly
        # classified positives (true positives)
        for bulk, final_class in final_bulk_classes.items():
            current_class = current_bulk_classes[bulk]
            if final_class is True:
                actual_positives += 1
                if current_class is True:
                    true_positives += 1

        # Calculate recall
        recall = true_positives / actual_positives
        return recall

    def _calculate_final_classes(self):
        '''
        Uses the true bulk values to classify each bulk as "good" or "not good"
        according to whether or not its bulk value quantile is above or below
        the `self.quantile_cutoff', respectively.

        Returns:
            final_bulk_classes  A dictionary whose values are the bulk ids and
                                whose values are Booleans. `True` means that
                                the bulk is above the threshold, and `False`
                                means that it is below.
        '''
        # Only need to calculate the final bulk classes once
        if not hasattr(self, 'final_bulk_classes'):

            # Only need to calculate the final bulk values once
            if not hasattr(self, 'final_bulk_values'):
                self.final_bulk_values = self.calculate_bulk_values(current=False)

            self.final_bulk_classes = self._classify_bulks(self.final_bulk_values)
        return self.final_bulk_classes

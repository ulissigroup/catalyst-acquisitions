'''
This submodule is meant to establish performance benchmarks for catalyst
discovery via screening of adsorption energies.
'''

__authors__ = ['Kevin Tran', 'Willie Neiswanger']
__emails__ = ['ktran@andrew.cmu.edu', 'willie@cs.cmu.edu']


import os
import warnings
from copy import deepcopy
import pickle
from pathlib import Path
import numpy as np
from scipy.stats import norm
from ..base import BaseActiveDiscoverer

# import line_profiler
# import atexit
# profile = line_profiler.LineProfiler()
# atexit.register(profile.print_stats)


class BaseAdsorptionDiscoverer(BaseActiveDiscoverer):
    '''
    Here we extend the `ActiveDiscovererBase` class while making the following
    assumptions:  1) we are trying to optimize the adsorption energy and 2) our
    inputs are a list of dictionaries with the 'energy' and 'std' keys.
    '''
    def __init__(self, model, quantile_cutoff, value_calculator,
                 training_features, training_labels, training_surfaces,
                 sampling_features, sampling_labels, sampling_surfaces,
                 n_samples=20, batch_size=200, init_train=True):
        '''
        Perform the initial training for the active discoverer, and then
        initialize all of the other attributes.

        Args:
            model               Instantiated model from the `.models` submodule
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
            value_calculator    A function that calculates the value of a
                                surface given the low-coverage adsorption
                                energy.
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
            batch_size          An integer indicating how many elements in the
                                sampling space you want to choose during each
                                batch of discovery. Defaults to 200.
            init_train          Boolean indicating whether or not you want to do
                                your first training during this initialization.
                                You should keep it `True` unless you're doing a
                                warm start (manually).
        '''
        # Sometimes our databases like to turn the Miller indices in our
        # `surface` arguments into numpy arrays when they should be tuples. Fix
        # that here.
        sampling_surfaces = [(mpid, tuple(index for index in miller), shift, top)
                             for mpid, miller, shift, top in sampling_surfaces]
        training_surfaces = [(mpid, tuple(index for index in miller), shift, top)
                             for mpid, miller, shift, top in training_surfaces]

        # Additional attributes for adsorption energies
        self.model = model
        self.value_calculator = value_calculator
        self.training_surfaces = deepcopy(training_surfaces)
        self.sampling_surfaces = deepcopy(sampling_surfaces)
        self.n_samples = n_samples

        # Error handling
        if not (0 < quantile_cutoff < 1):
            raise ValueError('The quantile cutoff should be between 0 and 1, '
                             'but is actually %.3f.' % quantile_cutoff)
        else:
            self.quantile_cutoff = quantile_cutoff

        # Used to save intermediate results
        self.cache_keys = {'training_features', 'training_labels', 'training_surfaces',
                           'sampling_features', 'sampling_labels', 'sampling_surfaces',
                           '_predicted_energies', 'residuals', 'uncertainties',
                           'reward_history', 'proxy_reward_history',
                           'batch_size', 'next_batch_number'}
        self.cache_affix = '_discovery_cache.pkl'
        Path(self.cache_location).mkdir(exist_ok=True)

        # Still want to do normal initialization
        super().__init__(training_features, training_labels,
                         sampling_features, sampling_labels,
                         batch_size=batch_size, init_train=init_train)

        # Cache some metadata
        _ = self._calculate_final_classes()

    def _train(self, next_batch):
        '''
        This method trains the model given the results from the
        `self._choose_next_batch` method. It also extends the new batch to the
        `training_*` properties of this class.

        Arg:
            next_batch  A 3-tuple containing a sequence for the features, a
                        sequence for the labels (DFT-calculated adsorption
                        energies in this case), and a sequence for the
                        surfaces.
        '''
        # Parse the incoming batch
        try:
            features, dft_energies, next_surfaces = next_batch
        # The `BaseDiscoverer` can do an initial training by feeding this
        # `_train` method two arguments:  features and labels. But this
        # `BaseAdsorptionDiscoverer` assumes that the `next_batch` argument
        # contains three arguments:  features, labels, AND surfaces. If we
        # don't get that third argument here, then assume that we are doing the
        # initial training and create it ourselves.
        except ValueError:
            assert self.next_batch_number == 0
            features, dft_energies = next_batch
            next_surfaces = deepcopy(self.training_surfaces)
            self.training_surfaces = []  # This should be popped

        # Get predictions and uncertainties for this next batch
        try:
            predictions, uncertainties = self.model.predict(features)
            residuals = predictions - dft_energies
            self.residuals.extend(residuals.tolist())
            self.uncertainties.extend(uncertainties)
        # If prediction doesn't work, then we probably haven't trained the
        # first batch. And if haven't done this, then there's no need to save
        # the residuals and uncertainty estimates.
        except AttributeError:
            pass

        # Extend new training points onto old ones
        self.training_features.extend(features)
        self.training_labels.extend(dft_energies)
        self.training_surfaces.extend(next_surfaces)

        # Retrain
        self.model.train(self.training_features, self.training_labels)
        self._save_current_run()

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
        self.reward_history.append(f_one)

    def _update_proxy_reward(self):
        '''
        This method updates a self.proxy_reward_history, which stores a proxy
        to our reward so far.
        '''
        # Calculate current bulk classes
        current_bulk_values = self.calculate_bulk_values(current=True)
        _, good_bulk_value_list = self._classify_bulks(current_bulk_values,
                                                       return_list=True)

        proxy_reward = np.min([val for (b, val) in good_bulk_value_list])
        self.proxy_reward_history.append(proxy_reward)

    def calculate_bulk_values(self, values_by_surface=None, current=True):
        '''
        Calculates the distributions of values of each bulk. Requires the
        `self.model` attribute to be able to accept `self.sampling_features`
        and return a 2-tuple of sequences:  the first sequence should be the
        predicted labels and the second sequence should be the predicted
        standard deviations/uncertainties.

        Args:
            values_by_surface   The output of the
                                `self.calculate_surface_values` method. It's
                                made explicit so that you can modify it if you
                                want. If `None`, it will call the method and
                                use the default output.
            current             A Boolean indicating whether you want the
                                "current" results or the final ones. "Current"
                                results come from data aggregations of already
                                "sampled" points with zero uncertainty and
                                "unsampled" points that whose means and
                                uncertainties are calculated by `self.model`.
                                So if this argument is `True`, then this will
                                return what the hallucination should think the
                                current state should be. If `False`, then this
                                will return the true results as per all the real
                                data fed into it. Unused if you supply the
                                `values_by_surface` argument.
        Returns:
            bulk_values     A dictionary whose keys are the bulk identifier
                            and whose values are a `np.array` of floats
                            indicating the "value" of each bulk.
        '''
        if values_by_surface is None:
            values_by_surface = self.calculate_surface_values(current=current)

        # Concatenate all the values for each surface onto their corresponding
        # bulks
        surface_values_by_bulk = {}
        for surface, values in values_by_surface.items():
            bulk_id = surface[0]
            surface_values = np.array(values).reshape((1, -1))
            try:
                surface_values_by_bulk[bulk_id] = np.concatenate((surface_values_by_bulk[bulk_id], surface_values), axis=0)
            except KeyError:
                surface_values_by_bulk[bulk_id] = surface_values

        # The value of a bulk is the average value of all of its surfaces
        bulk_values = {bulk_id: surface_values.mean(axis=0)
                       for bulk_id, surface_values in surface_values_by_bulk.items()}
        return bulk_values

    def calculate_surface_values(self, energies_by_surface=None, current=True):
        '''
        Calculates the "value" of each surface in the discovery space by
        assuming an Arrhenius-like relationship between the low coverage
        binding energy of the surface and its value.

        Args:
            energies_by_surface    The output of the
                                    `self.calculate_low_coverage_binding_energies_by_surface`
                                    method. Made an explicit argument so you
                                    can modify this argument if you want. If
                                    `None`, then it will call the method and
                                    grab defaults.
            current                 A Boolean indicating whether you want the
                                    "current" results or the final ones.
                                    "Current" results come from data
                                    aggregations of already "sampled" points
                                    with zero uncertainty and "unsampled"
                                    points that whose means and uncertainties
                                    are calculated by `self.model`. So if this
                                    argument is `True`, then this will return
                                    what the hallucination should think the
                                    current state should be.  If `False`, then
                                    this will return the true results as per
                                    all the real data fed into it. Unused if
                                    you supply the `energies_by_surface`
                                    argument.
        Returns:
            values_by_surface   A dictionary whose keys are a 4-tuple
                                containing surface information (mpid, miller,
                                shift, top) and whose values are a `np.array`
                                of floats indicating the "value" of a surface.
        '''
        if energies_by_surface is None:
            energies_by_surface = self.calculate_low_coverage_binding_energies_by_surface(current=current)

        values_by_surface = {}
        for surface, energies in energies_by_surface.items():
            values = [self.value_calculator(energy) for energy in energies]
            values_by_surface[surface] = values

        return values_by_surface

    def calculate_low_coverage_binding_energies_by_surface(self,
                                                           concatenated_energies=None,
                                                           current=True):
        '''
        Find/predicts the low coverage binding energies for each surface in the
        discovery space. Uses both DFT data (with zero uncertainty) and ML data
        (with predicted uncertainty).

        Args:
            concatenated_energies   The output of either the
                                    `self._concatenate_predicted_energies` or
                                    `self.concatenate_true_energies` methods.
                                    Or you can take them and modify them as you
                                    wish.
            current                 A Boolean indicating whether you want the
                                    "current" results or the final ones.
                                    "Current" results come from data
                                    aggregations of already "sampled" points
                                    with zero uncertainty and "unsampled"
                                    points that whose means and uncertainties
                                    are calculated by `self.model`. So if this
                                    argument is `True`, then this will return
                                    what the hallucination should think the
                                    current state should be.  If `False`, then
                                    this will return the true results as per
                                    all the real data fed into it. Unused if
                                    you supply the `concantenated_energies`
                                    argument.
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
        if concatenated_energies is None:
            if current is True:
                _, energies, stdevs, surfaces = self._concatenate_predicted_energies()
            elif current is False:
                _, energies, stdevs, surfaces = self._concatenate_true_energies()
            else:
                raise ValueError('The "current" argument should be Boolean, but is '
                                 'instead %s.' % type(current))
        else:
            _, energies, stdevs, surfaces = concatenated_energies

        # "Sample" all the sites `self.n_samples` times
        energies_by_surface = {}
        for energy, stdev, surface in zip(energies, stdevs, surfaces):
            try:
                samples = norm.rvs(loc=energy, scale=stdev, size=self.n_samples)
            # Sometimes stdev predictions go wrong. Move on if this happens
            except ValueError:
                assert np.isnan(stdev)
                warnings.warn('Just tried to sample site energy when std is '
                              'nan. Ignored and moving on.', RuntimeWarning)
                continue
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
            features    A list containing all the features of everything in the
                        discovery space (including both sampled and unsampled
                        sites).
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
        sampled_features = deepcopy(self.training_features)
        sampled_energies = deepcopy(self.training_labels)
        sampled_stdevs = [0. for _ in sampled_energies]
        sampled_surfaces = deepcopy(self.training_surfaces)

        # Use the model to make predictions on the unsampled space
        unsampled_features = deepcopy(self.sampling_features)
        try:
            predicted_energies, predicted_stdevs = self.model.predict(unsampled_features)
            unsampled_surfaces = deepcopy(self.sampling_surfaces)

            # Put it all together
            features = sampled_features + unsampled_features
            energies = sampled_energies + np.array(predicted_energies).tolist()
            stdevs = sampled_stdevs + np.array(predicted_stdevs).tolist()
            surfaces = sampled_surfaces + unsampled_surfaces

        # If there's nothing left to concatenate, then just return the already
        # sampled information
        except (np.core._exceptions.AxisError, RuntimeError):
            print('EXCEPTION: in adsorption_base > _concatenate_predicted_energies')
            energies, stdevs, surfaces = sampled_energies, sampled_stdevs, sampled_surfaces

        # ----------
        # Always return all surfaces
        surfaces = sampled_surfaces + unsampled_surfaces
        # ----------

        # Cache the energies
        self._predicted_energies = energies, stdevs, surfaces
        return features, energies, stdevs, surfaces

    def _concatenate_true_energies(self):
        '''
        This method will return the adsorption energies and corresponding
        "uncertainty estimates" on the entire discovery space. It will get data
        only from DFT results and always return an uncertainty of 0.

        Returns:
            features    A list containing all the features of everything in the
                        discovery space (including both sampled and unsampled
                        sites).
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
        sampled_features = deepcopy(self.training_features)
        sampled_energies = deepcopy(self.training_labels)
        sampled_stdevs = [0. for _ in sampled_energies]
        sampled_surfaces = deepcopy(self.training_surfaces)

        # Grab the results from the unsampled sites
        unsampled_features = deepcopy(self.sampling_features)
        unsampled_energies = deepcopy(self.sampling_labels)
        unsampled_stdevs = [0. for _ in unsampled_energies]
        unsampled_surfaces = deepcopy(self.sampling_surfaces)

        # Put it all together
        features = sampled_features + unsampled_features
        energies = np.concatenate((sampled_energies, unsampled_energies), axis=0)
        stdevs = np.array(sampled_stdevs + unsampled_stdevs)
        surfaces = sampled_surfaces + unsampled_surfaces
        return features, energies, stdevs, surfaces

    def _classify_bulks(self, bulk_values, return_list=False):
        '''
        Uses the true bulk values to classify each bulk as "good" or "not good"
        according to whether or not its bulk value quantile is above or below
        the `self.quantile_cutoff', respectively.

        Args:
            bulk_values     A dictionary whose keys are the bulk ids and whose
                            values are... the value of the bulk. Yeah this
                            naming convention isn't the best. See
                            `self.calculate_bulk_values`.
            return_list     If True, compute and return a list of (bulk, value)
                            for all bulks above the threshold.
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
                      for i, (bulk, value) in enumerate(sorted_bulks)}
        if return_list:
            good_bulk_value_list = [(bulk, value) for i, (bulk, value) in
                                    enumerate(sorted_bulks) if i <= cutoff]
            return good_bulks, good_bulk_value_list
        else:
            return good_bulks

    def _calculate_precision(self, current_bulk_classes):
        '''
        Calculates the precision of our binary classifier (see
        `self._update_reward`).

        Args:
            current_bulk_classes    The output of `self._classify_bulks` when
                                    you give in the current bulk values
        Returns:
            precision  The precision of our binary classifier (see
                       `self._update_reward`)
        '''
        # Initialize
        true_positives = 0
        false_positives = 0

        # Count the number of true positives and false positives
        for bulk, final_class in self.final_bulk_classes.items():
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

        Args:
            current_bulk_classes    The output of `self._classify_bulks` when
                                    you give in the current bulk values
        Returns:
            recall  The recall of our binary classifier (see
                    `self._update_reward`)
        '''
        # Initialize
        true_positives = 0
        actual_positives = 0

        # Count the number of actual positives and the number of correctly
        # classified positives (true positives)
        for bulk, final_class in self.final_bulk_classes.items():
            current_class = current_bulk_classes[bulk]
            if final_class is True:
                actual_positives += 1
                if current_class is True:
                    true_positives += 1

        # Calculate recall
        recall = true_positives / actual_positives
        return recall

    def _calculate_final_values(self):
        '''
        Return all bulk values, in finality (given all sites observed).

        Returns:
            final_bulk_values   A dictionary whose values are the bulk ids and
                                whose values are bulk values.
        '''
        # Only need to calculate the final bulk values once
        if not hasattr(self, 'final_bulk_values'):
            self.final_bulk_values = self.calculate_bulk_values(current=False)
        return self.final_bulk_values

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
            final_bulk_values = self._calculate_final_values()
            self.final_bulk_classes = self._classify_bulks(final_bulk_values)

        return self.final_bulk_classes

    def _pop_next_batch(self):
        '''
        Optional helper function that you can use to choose the next batch from
        `self.sampling_features`, remove it from the attribute, place the new
        batch onto the `self.training_features` attribute, increment the
        `self.next_batch_number`. Then do it all again for the
        `self.sampling_labels` and `self.training_labels` attributes.

        This method will only work if you have already sorted the
        `self.sampling_features` and `self.sampling_labels` such that the
        highest priority samples are earlier in the index.

        Returns:
            features    A list of length `self.batch_size` that contains the
                        next batch of features to train on.
            labels      A list of length `self.batch_size` that contains the
                        next batch of labels to train on.
        '''
        # The parent class will pop the features and labels just fine
        features, labels = super()._pop_next_batch()

        # We need to also pop the surfaces, too
        surfaces = []
        for _ in range(self.batch_size):
            try:
                surface = self.sampling_surfaces.pop(0)
                surfaces.append(surface)
            except IndexError:
                break
        return features, labels, surfaces

    def plot_performance(self, window=20, smoother='mean',
                         accuracy_units='eV', uncertainty_units='eV'):
        '''
        Light wrapper for plotting various performance metrics over the course
        of the discovery.

        Args:
            window              How many points to roll over during each
                                iteration
            smoother            String indicating how you want to smooth the
                                residuals over the course of the hallucination.
                                Corresponds exactly to the methods of the
                                `pandas.DataFrame.rolling` class, e.g., 'mean',
                                'median', 'min', 'max', 'std', 'sum', etc.
            accuracy_units      A string indicating the labeling units you want to
                                use for the accuracy figure.
            uncertainty_units   A string indicating the labeling units you want to
                                use for the uncertainty figure
        Returns:
            reward_fig      The matplotlib figure object for the reward plot
            accuracy_fig    The matplotlib figure object for the accuracy
            uncertainty_fig The matplotlib figure object for the uncertainty
            calibration_fig The matplotlib figure object for the calibration
            nll_fig         The matplotlib figure object for the negative log
                            likelihood
        '''
        return super().plot_performance(window=window, smoother=smoother,
                                        reward_name='Reward (F1 score)',
                                        accuracy_units=accuracy_units,
                                        uncertainty_units=uncertainty_units)

    def plot_predicted_vs_true_bulk_values(self):
        '''
        Plot of predictve bulk values versus true bulk values.

        Returns:
            fig     The matplotlib figure object for the learning curve
        '''
        # Initialize
        current_bulk_values = self.calculate_bulk_values(current=True)
        final_bulk_values = self._calculate_final_values()

        # This is very not elegant way of doing things, but I'm doing it
        pred_bulk_values, true_bulk_values = [], []
        for bulk, final_value in final_bulk_values.items():
            current_value = current_bulk_values[bulk]

            # Append predicted (current) and true (final) bulk values to lists
            pred_bulk_values.append(current_value)
            true_bulk_values.append(final_value)

            #pred_bulk_values.append(np.mean(current_value))
            #true_bulk_values.append(final_value[0])

        pred_bulk_values_mean = [np.mean(arr) for arr in pred_bulk_values]
        true_bulk_values_single = [arr[0] for arr in true_bulk_values]

        fig = super().plot_predicted_vs_true(pred_bulk_values_mean,
                                             true_bulk_values_single)
        fig = super().plot_predicted_vs_true_dist(pred_bulk_values,
                                                  true_bulk_values)
        return fig

    def _save_current_run(self):
        '''
        Cache the current point for (manual) warm-starts in case the last
        hallucination never finished. Should be called at the end of the
        `self._train` method.
        '''
        # Initialize predictions if we haven't made them yet
        if not hasattr(self, '_predicted_energies'):
            _ = self._concatenate_predicted_energies()

        # Save all the attributes-to-be-cached
        cache_name = (self.cache_location +
                      '%.3i%s' % (self.next_batch_number, self.cache_affix))
        cache = {key: getattr(self, key) for key in self.cache_keys}
        with open(cache_name, 'wb') as file_handle:
            pickle.dump(cache, file_handle)

        # Save the model state
        self.model.save()

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

        self.model.load()

    @property
    def cache_location(self):
        '''
        Uses the type of both the discoverer and the model to create a folder
        name to store caches in.
        '''
        return './' + type(self).__name__ + '_' + type(self.model).__name__ + '_caches/'

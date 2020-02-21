'''
This submodule is meant to establish performance benchmarks for catalyst
discovery via screening of adsorption energies.
'''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'


from copy import deepcopy
import numpy as np
from scipy.stats import norm
import seaborn as sns
import matplotlib.pyplot as plt
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
    def __init__(self, target_energy, training_features, training_labels,
                 sampling_features, sampling_labels,
                 batch_size=200, init_train=True):
        '''
        Perform the initial training for the active discoverer, and then
        initialize all of the other attributes.

        Args:
            target_energy       The optimal adsorption energy of an adsorption
                                site.
            training_features   A sequence that contains the features that can
                                be used to train/initialize the surrogate model
                                of the active discoverer.
            training_labels     A sequence that contains the labels that can
                                be used to train/initialize the surrogate model
                                of the active discoverer.
            sampling_features   A sequence containing the features for the rest
                                of the possible sampling space.
            sampling_labels     A sequence containing the labels for the rest
                                of the possible sampling space.
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
        '''
        # Find the current value
        reward = self.reward[-1]

        # Add the new reward
        for features, energy in zip(self.training_features, self.training_label):
            difference = energy - self.optimal_value
            reward += abs(difference)
        self.reward.append(reward)

    def plot_parity(self):
        '''
        Make the parity plot, where the residuals were the intermediate
        residuals we got along the way.

        Returns:
            fig     The matplotlib figure object for the parity plot
        '''
        # Pull the data that we have residuals for
        sampled_data = self.training_set[-len(self.residuals):]

        # Get both the DFT energies and the predicted energies
        energies_dft = np.array([doc['energy'] for doc in sampled_data])
        energies_pred = energies_dft + np.array(self.residuals)

        # Plot and format
        fig = plt.figure()
        energy_range = [min(energies_dft.min(), energies_pred.min()),
                        max(energies_dft.max(), energies_pred.max())]
        jgrid = sns.jointplot(energies_dft, energies_pred,
                              extent=energy_range*2,
                              kind='hex', bins='log')
        ax = jgrid.ax_joint
        _ = ax.set_xlabel('DFT-calculated adsorption energy [eV]')  # noqa: F841
        _ = ax.set_ylabel('ML-predicted adsorption energy [eV]')  # noqa: F841
        _ = fig.set_size_inches(10, 10)  # noqa: F841

        # Add the parity line
        _ = ax.plot(energy_range, energy_range, '--')  # noqa: F841
        return fig

    def calculate_bulk_values(self, n_samples=20, alpha=1, beta=1):
        '''
        Calculates the distributions of values of each bulk. Requires the
        `self.model` attribute to be able to accept `self.sampling_features`
        and return a 2-tuple of sequences:  the first sequence should be the
        predicted labels and the second sequence should be the predicted
        standard deviations/uncertainties.

        Args:
            n_samples   An integer indicating how many times we want to sample
                        all the distributions of values. We do this sampling
                        so that we can propagate uncertainty from the
                        site-level to the bulk-level.
            alpha       A float for the pre-exponential factor of the Arrhenius
                        relationship between energy and value
            beta        A float for the exponential factor of the Arrhenius
                        relationship between energy and value; akin to the
                        activation energy.
        Returns:
            bulk_values A dictionary whose
        '''
        values_by_surface = self.calculate_surface_values(n_samples, alpha, beta)

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

    def calculate_surface_values(self, n_samples=20, alpha=1, beta=1):
        '''
        Calculates the "value" of each surface in the discovery space by
        assuming an Arrhenius-like relationship between the low coverage
        binding energy of the surface and its value.

        Args:
            n_samples   An integer indicating how many times we want to sample
                        all the distributions of values. We do this sampling
                        so that we can propagate uncertainty from the
                        site-level to the bulk-level.
            alpha       A float for the pre-exponential factor of the Arrhenius
                        relationship between energy and value
            beta        A float for the exponential factor of the Arrhenius
                        relationship between energy and value; akin to the
                        activation energy.
        Returns:
            values_by_surface   A dictionary whose keys are a 4-tuple
                                containing surface information (mpid, miller,
                                shift, top) and whose values are a `np.array`
                                of floats indicating the "value" of a surface.
        '''
        energies_by_surface = self.calculate_low_coverage_binding_energies_by_surface(n_samples)

        # Perform an Arrhenius-like transformation of the binding energies to
        # get a rough estimate of value/activity.
        values_by_surface = {}
        for surface, energies in energies_by_surface.items():
            energy_diffs = np.abs(energies - self.target_energy)
            value = alpha * np.exp(-beta * energy_diffs)
            values_by_surface[surface] = value

        return values_by_surface

    def calculate_low_coverage_binding_energies_by_surface(self, n_samples=20):
        '''
        Find/predicts the low coverage binding energies for each surface in the
        discovery space. Uses both DFT data (with zero uncertainty) and ML data
        (with predicted uncertainty).

        Arg:
            n_samples   An integer indicating how many times we want to sample
                        all the distributions of energies. We do this sampling
                        so that we can propagate uncertainty from the
                        site-level to the bulk-level.
        Returns:
            low_cov_energies_by_surface     A dictionary whose keys are a
                                            4-tuple containing surface
                                            information (mpid, miller, shift,
                                            top) and whose values are a
                                            `np.array` of floats indicating the
                                            sampled low coverage adsorption
                                            energies of each surface.
        '''
        energies, stdevs, surfaces = self._concatenate_all_predictions()

        # "Sample" all the sites `n_samples` times
        energies_by_surface = {}
        for energy, stdev, surface in zip(energies, stdevs, surfaces):
            samples = np.array(norm.rvs(loc=energy, scale=stdev, size=n_samples)).reshape((1, -1))
            try:
                energies_by_surface[surface] = np.concatenate((energies_by_surface[surface], samples), axis=0)
            except KeyError:
                energies_by_surface[surface] = samples

        # Grab the lowest energy from each surface for each of the samples
        low_cov_energies_by_surface = {surface: sampled_energies.min(axis=0)
                                       for surface, sampled_energies in energies_by_surface.items()}
        return low_cov_energies_by_surface

    def _concatenate_all_predictions(self):
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


# def benchmark_adsorption_value(discoverers):
#     '''
#     This function will take the value curves of trained
#     `AdsorptionDiscovererBase` instances and the compare them to the floor and
#     ceilings of performance for you.
#
#     Arg:
#         discoverers     A dictionary whose keys are the string names that you
#                         want to label each discoverer with and whose values are
#                         the trained instances of an `AdsorptionDiscovererBase`
#                         where the `simulate_discovery` methods for each are
#                         already executed.
#     Returns:
#         value_fig  The matplotlib figure object for the value plot
#         axes        A dictionary whose keys correspond to the names of the
#                     discovery classes and whose values are the matplotlib axis
#                     objects.
#     '''
#     # Initialize
#     value_fig = plt.figure()
#     axes = {}
#
#     # Fetch the data used by the first discoverer to "train" the floor/ceiling
#     # models---i.e., get the baseline value histories.
#     example_discoverer_name = list(discoverers.keys())[0]
#     example_discoverer = list(discoverers.values())[0]
#     optimal_value = example_discoverer.optimal_value
#     sampling_size = example_discoverer.batch_size * len(example_discoverer.value_history)
#     training_docs = deepcopy(example_discoverer.training_set[:sampling_size])
#     sampling_docs = deepcopy(example_discoverer.training_set[-sampling_size:])
#
#     # Plot the worst-case scenario
#     random_discoverer = RandomAdsorptionDiscoverer(optimal_value,
#                                                    training_docs,
#                                                    sampling_docs)
#     random_discoverer.simulate_discovery()
#     sampling_sizes = [i*random_discoverer.batch_size
#                       for i, _ in enumerate(random_discoverer.value_history)]
#     random_label = 'random selection (worst case)'
#     axes[random_label] = plt.plot(sampling_sizes,
#                                   random_discoverer.value_history,
#                                   '--r',
#                                   label=random_label)
#
#     # Plot the value histories
#     for name, discoverer in discoverers.items():
#         sampling_sizes = [i*discoverer.batch_size
#                           for i, _ in enumerate(discoverer.value_history)]
#         ax = sns.scatterplot(sampling_sizes, discoverer.value_history, label=name)
#         axes[name] = ax
#
#     # Plot the best-case scenario
#     omniscient_discoverer = OmniscientAdsorptionDiscoverer(optimal_value,
#                                                            training_docs,
#                                                            sampling_docs)
#     omniscient_discoverer.simulate_discovery()
#     omniscient_label = 'omniscient selection (ideal)'
#     sampling_sizes = [i*omniscient_discoverer.batch_size
#                       for i, _ in enumerate(omniscient_discoverer.value_history)]
#     axes[omniscient_label] = plt.plot(sampling_sizes,
#                                       omniscient_discoverer.value_history,
#                                       '--b',
#                                       label=omniscient_label)
#
#     # Sort the legend correctly
#     legend_info = {label: handle for handle, label in zip(*ax.get_legend_handles_labels())}
#     labels = [random_label]
#     labels.extend(list(discoverers.keys()))
#     labels.append(omniscient_label)
#     handles = [legend_info[label] for label in labels]
#     ax.legend(handles, labels)
#
#     # Formatting
#     example_ax = axes[example_discoverer_name]
#     # Labels axes
#     _ = example_ax.set_xlabel('Number of discovery queries')  # noqa: F841
#     _ = example_ax.set_ylabel('Current value')  # noqa: F841
#     # Add commas to axes ticks
#     _ = example_ax.get_xaxis().set_major_formatter(FORMATTER)
#     _ = example_ax.get_yaxis().set_major_formatter(FORMATTER)
#     # Set bounds/limits
#     _ = example_ax.set_xlim([0, sampling_sizes[-1]])  # noqa: F841
#     _ = example_ax.set_ylim([0, random_discoverer.value_history[-1]])  # noqa: F841
#     # Set figure size
#     _ = value_fig.set_size_inches(15, 5)  # noqa: F841
#     return value_fig

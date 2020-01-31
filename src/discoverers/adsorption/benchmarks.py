'''
This submodule is meant to establish performance benchmarks for catalyst
discovery via screening of adsorption energies.
'''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'


import random
from copy import deepcopy
import numpy as np
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
    inputs are a list of dictionaries with the 'energy' key.
    '''
    def _update_regret(self):
        # Find the current regret
        regret = self.regret_history[-1]

        # Add the new regret
        for doc in self.training_batch:
            energy = doc['energy']
            difference = energy - self.optimal_value
            regret += abs(difference)
        self.regret_history.append(regret)

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


class RandomAdsorptionDiscoverer(AdsorptionDiscovererBase):
    '''
    This discoverer simply chooses new samples randomly. This is intended to be
    used as a baseline for active discovery.
    '''
    def _train(self):
        '''
        There's no real training here. We just do the bare minimum:  make up
        some residuals and extend the training set.
        '''
        try:
            # We'll just arbitrarily set the "model's" guess to the current average
            # of the training set
            energies = [doc['energy'] for doc in self.training_set]
            energy_guess = sum(energies) / max(len(energies), 1)
            residuals = [energy_guess - doc['energy'] for doc in self.training_batch]
            self.residuals.extend(residuals)

            # Mandatory extension of the training set to include this next batch
            self.training_set.extend(self.training_batch)

        # If this is the first batch, then don't bother recording residuals
        except TypeError:
            pass

    def _choose_next_batch(self):
        '''
        This method will choose a subset of the sampling space randomly and
        assign it to the `training_batch` attribute.. It will also remove
        anything we chose from the `self.sampling_space` attribute.
        '''
        random.shuffle(self.sampling_space)
        self._pop_next_batch()


class OmniscientAdsorptionDiscoverer(AdsorptionDiscovererBase):
    '''
    This discoverer has perfect knowledge of all data points and chooses the
    best ones perfectly. No method can beat this, and as such it provides a
    ceiling of performance.
    '''
    def _train(self):
        '''
        There's no real training here. We just do the bare minimum:  make up
        some residuals and extend the training set.
        '''
        try:
            # The model is omnipotent, so the residuals will be zero.
            residuals = [0.] * len(self.training_batch)
            self.residuals.extend(residuals)
            self.training_set.extend(self.training_batch)

        # If this is the first batch, then don't bother recording residuals
        except TypeError:
            pass

    def _choose_next_batch(self):
        '''
        This method will choose the portion of the sampling space whose
        energies are nearest to the target assign it to the `training_batch`
        attribute. It will also remove anything we chose from the
        `self.sampling_space` attribute.
        '''
        self.sampling_space.sort(key=lambda doc: abs(doc['energy'] - self.optimal_value))
        self._pop_next_batch()


def benchmark_adsorption_regret(discoverers):
    '''
    This function will take the regret curves of trained
    `AdsorptionDiscovererBase` instances and the compare them to the floor and
    ceilings of performance for you.

    Arg:
        discoverers     A dictionary whose keys are the string names that you
                        want to label each discoverer with and whose values are
                        the trained instances of an `AdsorptionDiscovererBase`
                        where the `simulate_discovery` methods for each are
                        already executed.
    Returns:
        regret_fig  The matplotlib figure object for the regret plot
        axes        A dictionary whose keys correspond to the names of the
                    discovery classes and whose values are the matplotlib axis
                    objects.
    '''
    # Initialize
    regret_fig = plt.figure()
    axes = {}

    # Fetch the data used by the first discoverer to "train" the floor/ceiling
    # models---i.e., get the baseline regret histories.
    example_discoverer_name = list(discoverers.keys())[0]
    example_discoverer = list(discoverers.values())[0]
    optimal_value = example_discoverer.optimal_value
    sampling_size = example_discoverer.batch_size * len(example_discoverer.regret_history)
    training_docs = deepcopy(example_discoverer.training_set[:sampling_size])
    sampling_docs = deepcopy(example_discoverer.training_set[-sampling_size:])

    # Plot the worst-case scenario
    random_discoverer = RandomAdsorptionDiscoverer(optimal_value,
                                                   training_docs,
                                                   sampling_docs)
    random_discoverer.simulate_discovery()
    sampling_sizes = [i*random_discoverer.batch_size
                      for i, _ in enumerate(random_discoverer.regret_history)]
    random_label = 'random selection (worst case)'
    axes[random_label] = plt.plot(sampling_sizes,
                                  random_discoverer.regret_history,
                                  '--r',
                                  label=random_label)

    # Plot the regret histories
    for name, discoverer in discoverers.items():
        sampling_sizes = [i*discoverer.batch_size
                          for i, _ in enumerate(discoverer.regret_history)]
        ax = sns.scatterplot(sampling_sizes, discoverer.regret_history, label=name)
        axes[name] = ax

    # Plot the best-case scenario
    omniscient_discoverer = OmniscientAdsorptionDiscoverer(optimal_value,
                                                           training_docs,
                                                           sampling_docs)
    omniscient_discoverer.simulate_discovery()
    omniscient_label = 'omniscient selection (ideal)'
    sampling_sizes = [i*omniscient_discoverer.batch_size
                      for i, _ in enumerate(omniscient_discoverer.regret_history)]
    axes[omniscient_label] = plt.plot(sampling_sizes,
                                      omniscient_discoverer.regret_history,
                                      '--b',
                                      label=omniscient_label)

    # Sort the legend correctly
    legend_info = {label: handle for handle, label in zip(*ax.get_legend_handles_labels())}
    labels = [random_label]
    labels.extend(list(discoverers.keys()))
    labels.append(omniscient_label)
    handles = [legend_info[label] for label in labels]
    ax.legend(handles, labels)

    # Formatting
    example_ax = axes[example_discoverer_name]
    # Labels axes
    _ = example_ax.set_xlabel('Number of discovery queries')  # noqa: F841
    _ = example_ax.set_ylabel('Cumulative regret [eV]')  # noqa: F841
    # Add commas to axes ticks
    _ = example_ax.get_xaxis().set_major_formatter(FORMATTER)
    _ = example_ax.get_yaxis().set_major_formatter(FORMATTER)
    # Set bounds/limits
    _ = example_ax.set_xlim([0, sampling_sizes[-1]])  # noqa: F841
    _ = example_ax.set_ylim([0, random_discoverer.regret_history[-1]])  # noqa: F841
    # Set figure size
    _ = regret_fig.set_size_inches(15, 5)  # noqa: F841
    return regret_fig

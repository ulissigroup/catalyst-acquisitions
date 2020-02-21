'''
This module is meant to be used to assess regression models and acquisition
functions for their fitness-for-use in our active discovery workflows.
'''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'


import sys
from abc import abstractmethod
import math
import warnings
from copy import deepcopy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import ticker

# The tqdm autonotebook is still experimental, and it warns us. We don't care,
# and would rather not hear about the warning everytime.
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from tqdm.autonotebook import tqdm


# Used to put commas into figures' axes' labels
FORMATTER = ticker.FuncFormatter(lambda x, p: format(int(x), ','))


class ActiveDiscovererBase:
    '''
    This is a parent class for simulating active discovery routines. The
    child classes are meant to be used to judge the efficacy of training
    ando/or acquisition routines. It does so by "simulating" what the routine
    would have done given a particular sampling space.
    '''
    def __init__(self, training_features, training_labels,
                 sampling_features, sampling_labels,
                 batch_size=200, init_train=True):
        '''
        Perform the initial training for the active discoverer, and then
        initialize all of the other attributes.

        Args:
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
        # Attributes we use to judge the discovery
        self.reward_history = [0.]
        self.residuals = []

        # Attributes we need to hallucinate the discovery
        self.next_batch_number = 0
        self.training_features = []
        self.training_features = list(deepcopy(training_features))
        self.training_labels = list(deepcopy(training_labels))
        self.sampling_features = list(deepcopy(sampling_features))
        self.sampling_labels = list(deepcopy(sampling_labels))
        self.batch_size = batch_size
        if init_train:
            self._train()

        # Attributes used in the `__assert_correct_hallucination` method
        self.__previous_training_set_len = len(self.training_features)
        self.__previous_sampling_space_len = len(self.sampling_features)
        self.__previous_reward_history_len = len(self.reward_history)
        self.__previous_residuals_len = len(self.residuals)

    def simulate_discovery(self, starting_batch_number=0):
        '''
        Perform the discovery simulation until all of the sampling space has
        been consumed.
        '''
        n_batches = math.ceil(len(self.sampling_features) / self.batch_size)
        for i in tqdm(range(0, n_batches), desc='Hallucinating discovery...'):
            self._hallucinate_next_batch()

    def _hallucinate_next_batch(self):
        '''
        Choose the next batch of data to get, add them to the `samples`
        attribute, and then re-train the surrogate model with the new samples.
        '''
        # Perform one iteration of active discovery
        self._choose_next_batch()
        self._train()
        self._update_reward()

        # Make sure it was done correctly
        self.__assert_correct_hallucination()

    @abstractmethod
    def _choose_next_batch(self):
        '''
        This method should:
            1. choose `self.batch_size` samples from the
               `self.sampling_features`  and `self.sampling_labels` attributes,
               and return them;
            2. it should also remove anything it selected from the
               `self.sampling_features` and `self.sampling_labels` attributes;
               and
            3. increment `self.next_batch_number` by 1
        '''
        pass

    @abstractmethod
    def _train(self):
        '''
        This method should take the output of the `choose_next_batch` method;
        calculate the current model's residuals on that batch and extend them
        onto the `residuals` attribute; use the training batch to [re]train the
        surrogate model; and finally extend the `self.training_features` attribute
        with the batch that it is passed.
        '''
        pass

    @abstractmethod
    def _update_reward(self):
        '''
        This method should take the output of the `choose_next_batch` method
        and then use it to calculate the new current reward. It should then
        append it to the `reward_history` attribute.
        '''
        pass

    def __assert_correct_hallucination(self):
        '''
        There are quite a few things that the user needs to do correctly to
        make a child class for this. This method will verify they're done
        correctly and then let the user know if it's not.
        '''
        # Make sure that the the sampling space is being reduced
        try:
            assert len(self.sampling_features) < self.__previous_sampling_space_len
            self.__previous_sampling_space_len = len(self.sampling_features)
        except AssertionError as error:
            message = ('\nWhen creating the `_choose_next_batch` method for '
                       'a child-class of `ActiveDiscovererBase`, you need to '
                       'remove the chosen batch from the `sampling_features` '
                       'attribute.')
            raise type(error)(str(error) + message).with_traceback(sys.exc_info()[2])

        # Make sure that the training set is being increased
        try:
            assert len(self.training_features) > self.__previous_training_set_len
            self.__previous_training_set_len = len(self.training_features)
        except AssertionError as error:
            message = ('\nWhen creating the `_train` method for a '
                       'child-class of `ActiveDiscovererBase`, you need to extend '
                       'the `training_features` attribute with the new training '
                       'batch.')
            raise type(error)(str(error) + message).with_traceback(sys.exc_info()[2])

        # Make sure that the residuals are being recorded
        try:
            assert len(self.residuals) > self.__previous_residuals_len
            self.__previous_residuals_len = len(self.residuals)
        except AssertionError as error:
            message = ('\nWhen creating the `_train` method for a '
                       'child-class of `ActiveDiscovererBase`, you need to extend '
                       'the `residuals` attribute with the model\'s residuals '
                       'of the new batch (before retraining).')
            raise type(error)(str(error) + message).with_traceback(sys.exc_info()[2])

        # Make sure we are calculating the reward at each step
        try:
            assert len(self.reward_history) > self.__previous_reward_history_len
            self.__previous_reward_history_len = len(self.reward_history)
        except AssertionError as error:
            message = ('\nWhen creating the `update_reward` method for a '
                       'child-class of `ActiveDiscovererBase`, you need to append '
                       'the `reward_history` attribute with the current '
                       'reward given the new batch.')
            raise type(error)(str(error) + message).with_traceback(sys.exc_info()[2])

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
        features = []
        labels = []
        for _ in range(self.batch_size):
            try:
                feature = self.sampling_features.pop(0)
                label = self.sampling_labels.pop(0)
                features.append(feature)
                labels.append(label)
            except IndexError:
                break
        self.next_batch_number += 1
        return features, labels

    def plot_performance(self, window=20, metric='mean'):
        '''
        Light wrapper for plotting the reward and residuals over the course of
        the discovery.

        Arg:
            window  How many residuals to average at each point in the learning
                    curve
            metric  String indicating which metric you want to plot in the
                    learning curve.  Corresponds exactly to the methods of the
                    `pandas.DataFrame.rolling` class, e.g., 'mean', 'median',
                    'min', 'max', 'std', 'sum', etc.
        Returns:
            reward_fig  The matplotlib figure object for the reward plot
            resid_fig   The matplotlib figure object for the residual plot
        '''
        reward_fig = self.plot_reward()
        learning_fig = self.plot_learning_curve(window)
        parity_fig = self.plot_parity()
        return reward_fig, learning_fig, parity_fig

    def plot_reward(self):
        '''
        Plot the reward vs. discovery batch

        Returns:
            fig     The matplotlib figure object for the reward plot
        '''
        # Plot
        fig = plt.figure()
        sampling_sizes = [i*self.batch_size for i, _ in enumerate(self.reward_history)]
        ax = sns.scatterplot(sampling_sizes, self.reward_history)

        # Format
        _ = ax.set_xlabel('Number of discovery queries')  # noqa: F841
        _ = ax.set_ylabel('Reward')  # noqa: F841
        _ = fig.set_size_inches(15, 5)  # noqa: F841
        _ = ax.get_xaxis().set_major_formatter(FORMATTER)
        _ = ax.get_yaxis().set_major_formatter(FORMATTER)
        return fig

    def plot_learning_curve(self, window=20, metric='mean'):
        '''
        Plot the rolling average of the residuals over time

        Arg:
            window  How many residuals to average at each point
            metric  String indicating which metric you want to plot.
                    Corresponds exactly to the methods of the
                    `pandas.DataFrame.rolling` class, e.g., 'mean', 'median',
                    'min', 'max', 'std', 'sum', etc.
        Returns:
            fig     The matplotlib figure object for the learning curve
        '''
        # Format the data
        df = pd.DataFrame(self.residuals, columns=['Residuals [eV]'])
        rolling_residuals = getattr(df, 'Residuals [eV]').rolling(window=window)
        rolled_values = getattr(rolling_residuals, metric)().values
        query_numbers = list(range(len(rolled_values)))

        # Create and format the figure
        fig = plt.figure()
        ax = sns.lineplot(query_numbers, rolled_values)
        _ = ax.set_xlabel('Number of discovery queries')
        _ = ax.set_ylabel('Rolling %s of residuals (window = %i) [eV]' % (metric, window))
        _ = ax.set_xlim([query_numbers[0], query_numbers[-1]])
        _ = fig.set_size_inches(15, 5)  # noqa: F841
        _ = ax.get_xaxis().set_major_formatter(FORMATTER)

        # Add a dashed line at zero residuals
        plt.plot([0, query_numbers[-1]], [0, 0], '--k')

        return fig

    @abstractmethod
    def plot_parity(self):
        '''
        This method should return an instance of a `matplotlib.pyplot.figure`
        object with the parity plotted.
        '''
        pass

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
import numpy as np
from scipy.stats import norm
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
FIG_SIZE = (9.5, 6.)


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
        self.uncertainties = []

        # Attributes we need to hallucinate the discovery
        self.next_batch_number = 0
        self.sampling_features = sampling_features
        self.sampling_labels = sampling_labels
        self.batch_size = batch_size
        if init_train is True:
            self.training_features = []
            self.training_labels = []
            self._train((training_features, training_labels))
        else:
            self.training_features = list(deepcopy(training_features))
            self.training_labels = list(deepcopy(training_labels))

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
        next_batch = self._choose_next_batch()
        self._train(next_batch)
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
    def _train(self, next_batch):
        '''
        This method should:
            1. take the output of the `choose_next_batch` method
            2. calculate the current model's residuals on that batch and extend
               them onto the `residuals` attribute
            3. calculate the current model's uncertainty estimates and extend
               them onto the `uncertainty_estimates` attributes
            4. use the training batch to [re]train the surrogate model
            5. extend the `self.training_features` attribute with the batch
               that it is passed.
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

    def plot_performance(self, window=20, smoother='mean',
                         accuracy_units='', uncertainty_units=''):
        '''
        Light wrapper for plotting various performance metrics over the course
        of the discovery.

        Arg:
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
            reward_fig  The matplotlib figure object for the reward plot
            resid_fig   The matplotlib figure object for the residual plot
        '''
        reward_fig = self.plot_reward(window=window, smoother=smoother)
        accuracy_fig = self.plot_accuracy(window=window, smoother=smoother,
                                          unit=accuracy_units)
        uncertainty_fig = self.plot_uncertainty_estimates(window=window, smoother=smoother,
                                                          unit=uncertainty_units)
        calibration_fig = self.plot_calibration(window=window, smoother=smoother)
        nll_fig = self.plot_nll(window=window, smoother=smoother)
        return reward_fig, accuracy_fig, uncertainty_fig, calibration_fig, nll_fig

    def plot_reward(self):
        '''
        Plot the reward vs. discovery batch number

        Returns:
            fig     The matplotlib figure object for the reward plot
        '''
        # Plot. Assume that the reward only updates per batch.
        fig = plt.figure()
        sampling_sizes = [i*self.batch_size for i, _ in enumerate(self.reward_history)]
        ax = sns.scatterplot(sampling_sizes, self.reward_history)

        # Format
        _ = ax.set_xlabel('Number of discovery queries')
        _ = ax.set_ylabel('Reward')
        _ = fig.set_size_inches(*FIG_SIZE)
        _ = ax.get_xaxis().set_major_formatter(FORMATTER)
        _ = ax.get_yaxis().set_major_formatter(FORMATTER)  # noqa: F841
        return fig

    @staticmethod
    def _plot_rolling_metric(metric_values, metric_name,
                             window=20, smoother='mean', unit=''):
        '''
        Helper function to plot model performance metrics across time in
        hallucination.

        Arg:
            metric_values   A sequence of floats that will be plotted against
                            batch number in the hallucination.
            metric_name     A string indicating what you want the values to be
                            labeled as in the plots.
            window          How many points to roll over during each iteration
            smoother        String indicating how you want to smooth the
                            residuals over the course of the hallucination.
                            Corresponds exactly to the methods of the
                            `pandas.DataFrame.rolling` class, e.g., 'mean',
                            'median', 'min', 'max', 'std', 'sum', etc.
            unit            [Optional] String indicating the units you want to
                            label the plot with
        Returns:
            fig     The matplotlib figure object for the learning curve
        '''
        # Format the data
        df = pd.DataFrame(metric_values, columns=[metric_name])
        rolling_residuals = getattr(df, metric_name).rolling(window=window)
        rolled_values = getattr(rolling_residuals, smoother)().values
        batch_numbers = list(range(len(rolled_values)))

        # Create and format the figure
        fig = plt.figure()
        ax = sns.lineplot(batch_numbers, rolled_values)
        _ = ax.set_xlabel('Number of discovery batches')
        if unit:
            unit = ' [' + unit + ']'
        _ = ax.set_ylabel('Rolling %s of %s%s' % (smoother, metric_name, unit))
        _ = ax.set_xlim([batch_numbers[0], batch_numbers[-1]])
        _ = fig.set_size_inches(*FIG_SIZE)
        _ = ax.get_xaxis().set_major_formatter(FORMATTER)  # noqa: F841
        return fig

    def plot_accuracy(self, window=20, smoother='mean', unit=''):
        '''
        Plot the the validation residuals as the hallucination progresses

        Arg:
            window      How many points to roll over during each iteration
            smoother    String indicating how you want to smooth the
                        residuals over the course of the hallucination.
                        Corresponds exactly to the methods of the
                        `pandas.DataFrame.rolling` class, e.g., 'mean',
                        'median', 'min', 'max', 'std', 'sum', etc.
            unit        [Optional] String indicating the units you want to
                        label the plot with
        Returns:
            fig     The matplotlib figure object for the learning curve
        '''
        fig = self._plot_rolling_metric(metric_values=np.abs(self.residuals),
                                        metric_name='|residuals|',
                                        window=window, smoother=smoother, unit=unit)
        return fig

    def plot_uncertainty_estimates(self, window=20, smoother='mean', unit=''):
        '''
        Plot the the validation uncertainties as the hallucination progresses

        Arg:
            window      How many points to roll over during each iteration
            smoother    String indicating how you want to smooth the
                        residuals over the course of the hallucination.
                        Corresponds exactly to the methods of the
                        `pandas.DataFrame.rolling` class, e.g., 'mean',
                        'median', 'min', 'max', 'std', 'sum', etc.
            unit        [Optional] String indicating the units you want to
                        label the plot with
        Returns:
            fig     The matplotlib figure object for the learning curve
        '''
        fig = self._plot_rolling_metric(metric_values=self.uncertainties,
                                        metric_name='predicted uncertainty (stdev)',
                                        window=window, smoother=smoother, unit=unit)
        return fig

    def plot_calibration(self, window=20, smoother='mean'):
        '''
        Plot the the model calibration as the hallucination progresses

        Arg:
            window      How many points to roll over during each iteration
            smoother    String indicating how you want to smooth the
                        residuals over the course of the hallucination.
                        Corresponds exactly to the methods of the
                        `pandas.DataFrame.rolling` class, e.g., 'mean',
                        'median', 'min', 'max', 'std', 'sum', etc.
        Returns:
            fig     The matplotlib figure object for the learning curve
        '''
        theoretical_cdfs = np.linspace(0, 1, 100)
        experimental_cdfs = [self.calculate_experimental_cdf(self.residuals,
                                                             self.uncertainties,
                                                             quantile)
                             for quantile in theoretical_cdfs]
        expected_calibration_errors = ((experimental_cdfs - theoretical_cdfs)**2).mean()
        fig = self._plot_rolling_metric(metric_values=expected_calibration_errors,
                                        metric_name='expected calibration error',
                                        window=window, smoother=smoother)
        return fig

    def plot_nll(self, window=20, smoother='mean'):
        '''
        Plot the the expected value of the model's negative-log-likelihood as
        the hallucination progresses

        Arg:
            window      How many points to roll over during each iteration
            smoother    String indicating how you want to smooth the
                        residuals over the course of the hallucination.
                        Corresponds exactly to the methods of the
                        `pandas.DataFrame.rolling` class, e.g., 'mean',
                        'median', 'min', 'max', 'std', 'sum', etc.
            unit        [Optional] String indicating the units you want to
                        label the plot with
        Returns:
            fig     The matplotlib figure object for the learning curve
        '''
        nlls = [-norm.logpdf(loc=resid, scale=std)
                for resid, std in zip(self.residuals, self.uncertainties)]
        fig = self._plot_rolling_metric(metric_values=nlls,
                                        metric_name='negative log likelihoods',
                                        window=window, smoother=smoother)
        return fig

    @staticmethod
    def calculate_experimental_cdf(residuals, uncertainties, quantile):
        '''
        Say we have the residuals of a model and the model's corresponding
        uncertainty estimates of each prediction (corresponding to the
        residuals). Let us assume that the uncertainty estimates are standard
        deviations for Gaussian-shaped distributions of residuals. We can
        compare the theoretical value for the cumulative distribution functions
        (CDFs) of Gaussian distributions with the experimental ones.

        This method will calculate the value of the of the experimental CDFs
        given the residuals, uncertainties, and theoretical CDF you want to
        compare it to.

        Args:
            residuals       A sequence of floats corresponding to the model's
                            residuals of predictions
            uncertanties    A sequence of floats correpsonding to the model's
                            estimate standard deviations for uncertainty. This
                            sequence should map directly with `residuals`.
            quantile        A float between [0, 1] that is used as the
                            "theoretical" CDF value for which to calculate the
                            experimental CDF.
        Returns:
            cdf     A float between [0, 1] indicating the experimental CDF
                    value
        '''
        # Normalize the residuals so they all should fall on the normal bell curve
        try:
            normalized_residuals = residuals.reshape(-1) / uncertainties.reshape(-1)
        except AttributeError:
            normalized_residuals = np.array(residuals).reshape(-1) / np.array(uncertainties).reshape(-1)

        # Count how many residuals fall inside here
        num_within_quantile = 0
        upper_bound = norm.ppf(quantile)
        for resid in normalized_residuals:
            if resid <= upper_bound:
                num_within_quantile += 1

        # Return the fraction of residuals that fall within the bounds
        cdf = num_within_quantile / len(residuals)
        return cdf

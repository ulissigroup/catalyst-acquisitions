'''
This submodule contains various utility functions we use to judge predictions.
'''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (mean_absolute_error,
                             mean_squared_error,
                             r2_score,
                             median_absolute_error)


def numpify(sequence):
    if isinstance(sequence, torch.Tensor):
        array = sequence.detach().cpu().numpy()
    elif isinstance(sequence, np.ndarray):
        array = sequence
    else:
        array = np.array(sequence)
    return array


def plot_errors(targets_pred, targets_actual, model_name):
    '''
    Args:
        targets_pred    Sequence of floats indicating the targets/labels you
                        were trying to predict
        targets_actual  Sequence of floats indicating the predictions you've
                        made for the targets
        model_name      String indicating the label you want for the model name
                        in the figure
    Returns:
        ax          `matplotlib.axes` object for the figure
        metrics     Dictionary containing various error metrics
    '''
    # Format input and calculate residuals
    targets_pred = numpify(targets_pred)
    targets_actual = numpify(targets_actual)
    residuals = targets_pred - targets_actual

    # Set plotting configs
    width = 7.5/3  # 1/3 of a page
    fontsize = 20
    rc = {'figure.figsize': (width, width),
          'font.size': fontsize,
          'axes.labelsize': fontsize,
          'axes.titlesize': fontsize,
          'xtick.labelsize': fontsize,
          'ytick.labelsize': fontsize,
          'legend.fontsize': fontsize}
    sns.set(rc=rc)
    sns.set_style('ticks')

    # Plot
    lims = [-4, 4]
    grid = sns.jointplot(targets_actual.reshape(-1), targets_pred,
                         kind='hex',
                         bins='log',
                         extent=lims+lims)
    ax = grid.ax_joint
    _ = ax.set_xlim(lims)
    _ = ax.set_ylim(lims)
    _ = ax.plot(lims, lims, '--')
    _ = ax.set_xlabel('DFT $\Delta$E [eV]')  # noqa: W605
    _ = ax.set_ylabel('%s $\Delta$E [eV]' % model_name)  # noqa: W605

    # Calculate the error metrics
    mae = mean_absolute_error(targets_actual, targets_pred)
    rmse = np.sqrt(mean_squared_error(targets_actual, targets_pred))
    mdae = median_absolute_error(targets_actual, targets_pred)
    marpd = np.abs(2 * residuals /
                   (np.abs(targets_pred) + np.abs(targets_actual))
                   ).mean() * 100
    r2 = r2_score(targets_actual, targets_pred)
    ppmcc = np.corrcoef(targets_actual.reshape(-1), targets_pred)[0, 1]

    # Report
    text = ('  MDAE = %.2f eV\n' % mdae +
            '  MAE = %.2f eV\n' % mae +
            '  RMSE = %.2f eV\n' % rmse +
            '  MARPD = %i%%\n' % marpd)
    _ = ax.text(x=lims[0], y=lims[1], s=text,
                horizontalalignment='left',
                verticalalignment='top',
                fontsize=fontsize)
    plt.show()

    # Parse output
    metrics = {'mae': mae,
               'rmse': rmse,
               'mdae': mdae,
               'marpd': marpd,
               'r2': r2,
               'ppmcc': ppmcc}
    return ax, metrics

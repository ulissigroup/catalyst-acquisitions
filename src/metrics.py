'''
This submodule contains various utility functions we use to judge predictions.
'''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'

import numpy as np
from scipy.stats import norm
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import Polygon, LineString
from shapely.ops import polygonize, unary_union
from sklearn.metrics import (mean_absolute_error,
                             mean_squared_error,
                             r2_score,
                             median_absolute_error)


# Import the correct TQDM depending on where we are
try:
    shell = get_ipython().__class__.__name__
    if shell == "ZMQInteractiveShell":
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
except NameError:
    from tqdm import tqdm


def plot_metrics(targets_pred, targets_actual, stdevs, model_name):
    '''
    Args:
        targets_pred    Sequence of floats indicating the targets/labels you
                        were trying to predict
        targets_actual  Sequence of floats indicating the predictions you've
                        made for the targets
        stdevs          Sequence of floats indicating the corresponding
                        uncertainties for each of the predictions
        model_name      String indicating the label you want for the model name
                        in the figure
    Returns:
        figs        Dictionary containing various matplotlib objects of
                    different plots
        metrics     Dictionary containing various model performance metrics
    '''
    # Format input and calculate residuals
    targets_pred = _numpify(targets_pred)
    targets_actual = _numpify(targets_actual)
    stdevs = _numpify(stdevs)
    residuals = targets_pred - targets_actual

    # Make plots & calculate metrics
    error_ax, error_metrics = plot_errors(targets_pred, targets_actual, residuals, model_name)
    calibration_fig, cal_metrics = plot_calibration(residuals, stdevs, model_name)
    sharpness_ax, sharpness_metrics = plot_sharpness(stdevs, model_name)

    # Concatenate results
    figs = {'error axes': error_ax,
            'calibration figure': calibration_fig,
            'sharpness axes': sharpness_ax}
    metrics = {'nll': calculate_nll(residuals, stdevs)}
    for dict_ in [error_metrics, cal_metrics, sharpness_metrics]:
        for key, value in dict_.items():
            metrics[key] = value

    return figs, metrics


def plot_errors(targets_pred, targets_actual, residuals, model_name):
    '''
    Args:
        targets_pred    Sequence of floats indicating the targets/labels you
                        were trying to predict
        targets_actual  Sequence of floats indicating the predictions you've
                        made for the targets
        residuals       Difference between targets_pred and targets_actual
        model_name      String indicating the label you want for the model name
                        in the figure
    Returns:
        ax          `matplotlib.axes` object for the figure
        metrics     Dictionary containing various error metrics
    '''
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

    # Parse output
    metrics = {'mae': mae,
               'rmse': rmse,
               'mdae': mdae,
               'marpd': marpd,
               'r2': r2,
               'ppmcc': ppmcc}
    return ax, metrics


def _numpify(sequence):
    if isinstance(sequence, torch.Tensor):
        array = sequence.detach().cpu().numpy()
    elif isinstance(sequence, np.ndarray):
        array = sequence
    else:
        array = np.array(sequence)
    return array


def plot_calibration(residuals, stdevs, model_name):
    '''
    Args:
        residuals       Difference between targets_pred and targets_actual
        stdevs          Sequence of floats indicating the corresponding
                        uncertainties for each of the predictions
        model_name      String indicating the label you want for the model name
                        in the figure
    Returns:
        fig         `matplotlib.figure.Figuer` object for the figure
        metrics     Dictionary containing various calibration metrics
    '''
    # Calculate the calibration error
    predicted_pi = np.linspace(0, 1, 100)
    observed_pi = [_calculate_density(residuals, stdevs, quantile)
                   for quantile in tqdm(predicted_pi, desc='Calibration')]
    calibration_error = ((predicted_pi - observed_pi)**2).sum()

    # Figure settings
    width = 4  # Because it looks good
    fontsize = 12
    rc = {'figure.figsize': (width, width),
          'font.size': fontsize,
          'axes.labelsize': fontsize,
          'axes.titlesize': fontsize,
          'xtick.labelsize': fontsize,
          'ytick.labelsize': fontsize,
          'legend.fontsize': fontsize}
    sns.set(rc=rc)
    sns.set_style('ticks')
    figsize = (4, 4)

    # Plot the calibration curve
    fig_cal = plt.figure(figsize=figsize)
    ax_ideal = sns.lineplot([0, 1], [0, 1], label='ideal')
    _ = ax_ideal.lines[0].set_linestyle('--')
    _ = sns.lineplot(predicted_pi, observed_pi, label=model_name)
    _ = plt.fill_between(predicted_pi, predicted_pi, observed_pi,
                         alpha=0.2, label='miscalibration area')
    _ = ax_ideal.set_xlabel('Expected cumulative distribution')
    _ = ax_ideal.set_ylabel('Observed cumulative distribution')
    _ = ax_ideal.set_xlim([0, 1])
    _ = ax_ideal.set_ylim([0, 1])

    # Calculate the miscalibration area.
    polygon_points = []
    for point in zip(predicted_pi, observed_pi):
        polygon_points.append(point)
    for point in zip(reversed(predicted_pi), reversed(predicted_pi)):
        polygon_points.append(point)
    polygon_points.append((predicted_pi[0], observed_pi[0]))
    polygon = Polygon(polygon_points)
    x, y = polygon.exterior.xy  # original data
    ls = LineString(np.c_[x, y])  # closed, non-simple
    lr = LineString(ls.coords[:] + ls.coords[0:1])
    mls = unary_union(lr)
    polygon_area_list = [poly.area for poly in polygonize(mls)]
    miscalibration_area = np.asarray(polygon_area_list).sum()

    # Annotate the plot with the miscalibration area
    plt.text(x=0.95, y=0.05,
             s='Miscalibration area = %.3f' % miscalibration_area,
             verticalalignment='bottom',
             horizontalalignment='right',
             fontsize=fontsize)

    metrics = {'calibration error': calibration_error,
               'miscalibration area': miscalibration_area}
    return fig_cal, metrics


def _calculate_density(residuals, stdevs, percentile):
    '''
    Calculate the fraction of the residuals that fall within the lower
    `percentile` of their respective Gaussian distributions, which are
    defined by their respective uncertainty estimates.
    '''
    # Find the normalized bounds of this percentile
    upper_bound = norm.ppf(percentile)

    # Normalize the residuals so they all should fall on the normal bell curve
    normalized_residuals = residuals.reshape(-1) / stdevs.reshape(-1)

    # Count how many residuals fall inside here
    num_within_quantile = 0
    for resid in normalized_residuals:
        if resid <= upper_bound:
            num_within_quantile += 1

    # Return the fraction of residuals that fall within the bounds
    density = num_within_quantile / len(residuals)
    return density


def plot_sharpness(stdevs, model_name):
    '''
    Args:
        stdevs          Sequence of floats indicating the corresponding
                        uncertainties for each of the predictions
        model_name      String indicating the label you want for the model name
                        in the figure
    Returns:
        ax          `matplotlib.axes` object for the figure
        metrics     Dictionary containing various calibration metrics
    '''
    # Figure settings
    xlim = [0., 1.]
    figsize = (4, 4)
    fontsize = 12
    _ = plt.figure(figsize=figsize)
    ax_sharp = sns.distplot(stdevs, kde=False, norm_hist=True)
    ax_sharp.set_xlim(xlim)
    ax_sharp.set_xlabel('Predicted standard deviations (eV)')
    ax_sharp.set_ylabel('Normalized frequency')
    ax_sharp.set_yticklabels([])
    ax_sharp.set_yticks([])

    # Calculate and report sharpness
    sharpness = np.sqrt(np.mean(stdevs**2))
    _ = ax_sharp.axvline(x=sharpness, label='sharpness')
    dispersion = np.sqrt(((stdevs - stdevs.mean())**2).sum() / (len(stdevs)-1)) / stdevs.mean()
    if sharpness < (xlim[0] + xlim[1]) / 2:
        text = '\n  Sharpness = %.2f eV\n  C$_v$ = %.2f' % (sharpness, dispersion)
        h_align = 'left'
    else:
        text = '\nSharpness = %.2f eV  \nC$_v$ = %.2f  ' % (sharpness, dispersion)
        h_align = 'right'
    _ = ax_sharp.text(x=sharpness, y=ax_sharp.get_ylim()[1],
                      s=text,
                      verticalalignment='top',
                      horizontalalignment=h_align,
                      fontsize=fontsize)

    metrics = {'sharpness': sharpness,
               'dispersion': dispersion}
    return ax_sharp, metrics


def calculate_nll(residuals, stdevs):
    '''
    Args:
        residuals   Difference between targets_pred and targets_actual
        stdevs      Sequence of floats indicating the corresponding
                    uncertainties for each of the predictions
    Returns:
        nll     Negative-log-likelihood of observing the residuals given the
                predicted stdevs
    '''
    nll_list = []
    for (res, std) in zip(residuals, stdevs):
        nll_list.append(norm.logpdf(res, scale=std))
    nll = -1 * np.sum(nll_list)
    return nll

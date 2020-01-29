"""
Code for simulating (part of) the generative process of our model:
propagating uncertainty from the predictive distribution over the site energy
values, through value of the surface, and finally to the value of the bulk.
"""

import numpy as np


def get_normal_params_dict(samp_array):
    """Compute and return dict of Gaussian params."""
    normal_params_dict = {'loc': np.mean(samp_array),
                          'scale': np.std(samp_array)}
    return normal_params_dict

def get_distrib_over_min_sites(site_value_distrib_list, distrib_type='samples',
                               sim_type='samples', n_samples=1e5,
                               return_type='samples'):
    """
    Return distribution over the minimum of site values.

    Args:
        site_value_distrib_list: list of distributions over site values, each
                                 element corresponding to a dfferent site.
        distrib_type: format of distrib in distrib_list, could be 'params' or
                      'samples'.
        sim_type: type of simuation to do, could be 'samples' or 'analytic'
        n_samples: number of samples to generate (for 'samples' sim_type)
        return_type: format of return distrib, could be 'params' or 'samples'
    """
    if sim_type == 'samples':
        
        # Compute samp_list (for each site)
        if distrib_type == 'params':
            samp_list = [np.random.normal(loc=d['loc'], scale=d['scale'],
                                          size=(n_samples,))
                         for d in site_value_distr_list]
        else:
            samp_list = [d.reshape(-1) for d in site_value_distr_list]

        # Compute mins array
        min_samp = np.min(np.array(samp_list), 0)

        # Return mins array or dict of params
        if return_type == 'params':
            return get_normal_params_dict(min_samp)
        else:
           return min_samp


def get_distrib_over_surface_value(min_site_distrib, optimal_min_site_value,
                                   distrib_type='params', n_samples=1e5,
                                   return_type='samples'):
    """
    Return distribution over the value of a surface. This distribution depends
    on the distribution over the minimum value of sites on a surface and on the
    "optimal" minimum value of the sites.

    Args:
        min_site_distrib: distribution over the minimum site value on surface.
        optimal_min_site_value: optimal minimum site value on surface
        distrib_type: format of distrib for min_site_distrib, could be 'params' or
                      'samples'.
        n_samples: number of samples to generate (for 'samples' sim_type)
        return_type: format of return distrib, could be 'params' or 'samples'
    """
    # Convert to samples
    if distribe_type == 'params':
        min_site_samp = np.random.normal(loc=min_site_distrib['loc'],
                                         scale=min_site_distrib['scale'],
                                         shape=(n_samples,))
    else:
        min_site_samp = min_site_distrib.reshape(-1,)

    # Compute each sample: exp(|s - opt_min_site|) for s in min_site_samp
    surface_value_samp = np.exp(np.abs(min_site_samp - optimal_min_site_value))

    # Return samp array or dict of params
    if return_type == 'params':
        return get_normal_params_dict(surface_value_samp)
    else:
       return surface_value_samp


def get_distrib_over_bulk_value(surface_value_distrib_list):
    """
    Returns distribution over the value of a bulk. This distribution depends on
    the distribution over the values of surfaces in the bulk.
    """
    pass

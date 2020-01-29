"""
Code for simulating (part of) the generative process of our model:
propagating uncertainty from the predictive distribution over the site energy
values, through value of the surface, and finally to the value of the bulk.
"""

import numpy as np



def get_distrib_over_min_sites(site_value_distrib_list, distrib_type='params',
                               sim_type='samples', n_samples=1e5): 
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
        
        # compute samp_list (for each site)
        if distrib_type == 'params':
            samp_list = [np.random.normal(loc=d['loc'], scale=d['scale'],
                                          size=(n_samples,))
                         for d in site_value_distr_list]

        # compute mins array
        min_arr = np.min(np.array(samp_list), 0)

        # return mins array or dict of params
        if return_type == 'params':
            return {'loc': np.mean(min_arr), 'scale': np.std(min_arr)}
        else:
           return min_arr


def get_distrib_over_surface_value(min_site_distrib, optimal_min_site_value,
                                   distrib_type='params'):
    """
    Return distribution over the value of a surface. This distribution depends
    on the distribution over the minimum value of sites on a surface and on the
    "optimal" minimum value of the sites.

    Args:
        min_site_distrib: distribution over the minimum site value on surface.
        optimal_min_site_value: optimal minimum site value on surface
        distrib_type: format of distrib for min_site_distrib, could be 'params' or
                      'samples'.
    """
    if distribe_type == 'params':
        pass


def get_distrib_over_bulk_value(surface_value_distrib_list):
    """
    Returns distribution over the value of a bulk. This distribution depends on
    the distribution over the values of surfaces in the bulk.
    """
    pass

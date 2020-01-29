"""
Code for simulating (part of) the generative process of our model:
propagating uncertainty from the predictive distribution over the site energy
values, through value of the surface, and finally to the value of the bulk.
"""

import numpy as np



def get_distrib_over_min_sites(site_value_distrib_list): 
    """
    Return distribution over the minimum of site values.

    Args:
       site_value_distrib_list: list of distributions over site values, each
                                element corresponding to a dfferent site.
    """
    pass


def get_distrib_over_surface_value(min_site_distrib, optimal_min_site_value):
    """
    Return distribution over the value of a surface. This distribution depends
    on the distribution over the minimum value of sites on a surface and on the
    "optimal" minimum value of the sites.

    Args:
        min_site_distrib: distribution over the minimum site value on surface.
        optimal_min_site_value: optimal minimum site value on surface
    """
    pass


def get_distrib_over_bulk_value(surface_value_distrib_list):
    """
    Returns distribution over the value of a bulk. This distribution depends on
    the distribution over the values of surfaces in the bulk.
    """
    pass

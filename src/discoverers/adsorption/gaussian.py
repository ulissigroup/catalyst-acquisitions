'''
This submodule houses the `GaussianDiscoverer` child class of
`BaseAdsorptionDiscoverer` that hallucinates the performance of an incumbent
method that uses a model's prediction to perform a sort of Thompson sampling in
the context of discovering catalysts by screening their adsorption energies. We
call it `GaussianDiscoverer` because we choose sample sites around a target
energy using a Gaussian distribution.

Refer to https://www.nature.com/articles/s41929-018-0142-1 for more details.
'''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'


import random
from bisect import bisect_right
import numpy as np
from scipy.stats import norm
from .base import BaseAdsorptionDiscoverer


class GaussianDiscoverer(BaseAdsorptionDiscoverer):
    '''
    This discoverer uses a Gaussian selection method with a TPOT model to select
    new sampling points.

    ...sorry for the awful code. This is a hack-job and I know it.
    '''

    def __init__(self, target_energy, assumed_stdev=0.1, *args, **kwargs):
        '''
        In addition to the normal things that this class's parent classes do in
        `__init__` this method also instantiates the `TPOTWrapper`
        '''
        self.target_energy = target_energy
        self.assumed_stdev = assumed_stdev
        super().__init__(*args, **kwargs)

    def _choose_next_batch(self):
        '''
        Choose the next batch "randomly", where the probability of selecting
        sites are weighted using a combination of a Gaussian distribution and
        TPOT's prediction of their distance from the optimal energy.

        Returns:
            features    The features that this method chose to investigate next
            labels      The labels that this method chose to investigate next
            surfaces    The surfaces that this method chose to investigate next
        '''
        # Use the energies to calculate probabilities of selecting each site
        energies, _ = self.model.predict(self.sampling_features)
        gaussian_distribution = norm(loc=self.target_energy, scale=self.assumed_stdev)
        probability_densities = [gaussian_distribution.pdf(energy) for energy in energies]

        # Perform a weighted shuffling of the sampling space such that sites
        # with better energies are more likely to be early in the list
        features, labels, surfaces = self.weighted_shuffle(self.sampling_features,
                                                           self.sampling_labels,
                                                           self.sampling_surfaces,
                                                           weights=probability_densities)
        self.sampling_features = features
        self.sampling_labels = labels
        self.sampling_surfaces = [tuple(surface) for surface in surfaces]

        # Now that the samples are sorted, find the next ones and add them to
        # the training set
        features, labels, surfaces = self._pop_next_batch()
        return features, labels, surfaces

    @staticmethod
    def weighted_shuffle(*sequences, weights):
        '''
        This function will shuffle a sequence using weights to increase the chances
        of putting higher-weighted elements earlier in the list. Credit goes to
        Nicky Van Foreest, whose function I based this off of.

        Args:
            sequence    Any number of sequences of elements that you want
                        shuffled
            weights     A sequence that is the same length as the `sequence`
                        that contains the corresponding probability weights for
                        selecting/choosing each element in `sequence`
        Returns:
            shuffled_list   A list whose elements are identical to those in the
                            `sequence` argument, but randomly shuffled such
                            that the elements with higher weights are more
                            likely to be in the front/start of the list.
        '''
        shuffled_arrays = [np.empty_like(sequence) for sequence in sequences]

        # Pack together the elements in the sequences and their respective weights
        packets = list(zip(*sequences, weights))
        for i in range(len(packets)):

            # Randomly choose one of the elements, and get the corresponding index
            cumulative_weights = np.cumsum([packet[-1] for packet in packets])
            rand = random.random() * cumulative_weights[-1]
            selected_index = bisect_right(cumulative_weights, rand)

            # Pop the element out so we don't re-select
            packet = packets.pop(selected_index-1)

            # Don't need to save the last item in each packet, which is the
            # weight
            for j, value in enumerate(list(packet)[:-1]):
                shuffled_arrays[j][i] = value

        return (array.tolist() for array in shuffled_arrays)

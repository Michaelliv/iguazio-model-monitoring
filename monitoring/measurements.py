from dataclasses import dataclass

import numpy as np


@dataclass
class TotalVarianceDistance:
    """
    Provides a symmetric drift distance between two periods t and u
    Z - vector of random variables
    Pt - Probability distribution over time span t
    """

    distrib_t: np.ndarray
    distrib_u: np.ndarray

    def compute(self) -> float:
        return np.sum(np.abs(self.distrib_t - self.distrib_u)) / 2


@dataclass
class HellingerDistance:
    """
    Hellinger distance is an f divergence measure, similar to the Kullback-Leibler (KL) divergence.
    However, unlike KL Divergence the Hellinger divergence is symmetric and bounded over a probability space.
    """

    distrib_t: np.ndarray
    distrib_u: np.ndarray

    def compute(self) -> float:
        return np.sqrt(
            0.5 * ((np.sqrt(self.distrib_u) - np.sqrt(self.distrib_t)) ** 2).sum()
        )


@dataclass
class KullbackLeiblerDivergence:
    """
    KL Divergence (or relative entropy) is a measure of how one probability distribution differs from another.
    It is an asymmetric measure (thus it's not a metric) and it doesn't satisfy the triangle inequality.
    KL Divergence of 0, indicates two identical distributions.
    """

    distrib_t: np.ndarray
    distrib_u: np.ndarray

    def compute(self, capping=None, kld_scaling=0.0001) -> float:
        t_u = np.sum(
            np.where(
                self.distrib_t != 0,
                (self.distrib_t)
                * np.log(
                    self.distrib_t
                    / np.where(self.distrib_u != 0, self.distrib_u, kld_scaling)
                ),
                0,
            )
        )
        u_t = np.sum(
            np.where(
                self.distrib_u != 0,
                (self.distrib_u)
                * np.log(
                    self.distrib_u
                    / np.where(self.distrib_t != 0, self.distrib_t, kld_scaling)
                ),
                0,
            )
        )
        result = t_u + u_t
        if capping:
            return capping if result == float("inf") else result
        return result

from dataclasses import dataclass

import pandas as pd
from numpy import sum, where
from numpy.core._multiarray_umath import sqrt, power, log


@dataclass
class TotalVarianceDistance:
    """
    Provides a symmetric drift distance between two periods t and u
    Z - vector of random variables
    Pt - Probability distribution over time span t
    """

    distrib_t: pd.Series
    distrib_u: pd.Series

    def compute(self) -> float:
        return sum(abs(self.distrib_t - self.distrib_u)) / 2


@dataclass
class HellingerDistance:
    """
    Hellinger distance is an f divergence measure, similar to the Kullback-Leibler (KL) divergence.
    However, unlike KL Divergence the Hellinger divergence is symmetric and bounded over a probability space.
    """

    distrib_t: pd.Series
    distrib_u: pd.Series

    def compute(self) -> float:
        dividend = sqrt(sum(power(sqrt(self.distrib_t) - sqrt(self.distrib_u), 2)))
        divisor = sqrt(2)
        return dividend / divisor


@dataclass
class KullbackLeiblerDivergence:
    """
    KL Divergence (or relative entropy) is a measure of how one probability distribution differs from another.
    It is an asymmetric measure (thus it's not a metric) and it doesn't satisfy the triangle inequality.
    KL Divergence of 0, indicates two identical distributions.
    """

    distrib_t: pd.Series
    distrib_u: pd.Series

    def compute(self) -> float:
        t_u = sum(
            where(
                self.distrib_t != 0,
                self.distrib_t * log(self.distrib_t / self.distrib_t),
                0,
            )
        )
        u_t = sum(
            where(
                self.distrib_u != 0,
                self.distrib_u * log(self.distrib_u / self.distrib_u),
                0,
            )
        )
        return t_u + u_t

# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import numpy as np
import math
from scipy.stats import truncnorm


class AbstractArm(object):
    def __init__(self, mean, variance, random_state):
        """
        Args:
            mean: expectation of the arm
            variance: variance of the arm
            random_state (int): seed to make experiments reproducible
        """
        self.mean = mean
        self.variance = variance

        self.local_random = np.random.RandomState(random_state)

    def sample(self):
        pass


class ArmTruncNorm():
    def __init__(self, original_mean=0, a=-1, b=1, original_std=0.1):
        a, b = (a - original_mean) / original_std, (b - original_mean) / original_std
        self.a = a
        self.b = b
        self.true_sigma = original_std
        self.true_mean = original_mean
        self.mean, self.sigma = truncnorm.stats(a=self.a, b=self.b, loc=self.true_mean, scale=self.true_sigma)

    def sample(self):
        return truncnorm.rvs(a=self.a, b=self.b, loc=self.true_mean, scale=self.true_sigma)


class ArmBernoulli(AbstractArm):
    def __init__(self, p, random_state=0):
        """
        Bernoulli arm
        Args:
             p (float): mean parameter
             random_state (int): seed to make experiments reproducible
        """
        self.p = p
        super(ArmBernoulli, self).__init__(mean=p,
                                           variance=p * (1. - p),
                                           random_state=random_state)

    def sample(self):
        return self.local_random.rand(1) < self.p


class ArmBeta(AbstractArm):
    def __init__(self, a, b, random_state=0):
        """
        arm having a Beta distribution
        Args:
             a (float): first parameter
             b (float): second parameter
             random_state (int): seed to make experiments reproducible
        """
        self.a = a
        self.b = b
        super(ArmBeta, self).__init__(mean=a / (a + b),
                                      variance=(a * b) / ((a + b) ** 2 * (a + b + 1)),
                                      random_state=random_state)

    def sample(self):
        return self.local_random.beta(self.a, self.b, 1)


class ArmExp(AbstractArm):
    # https://en.wikipedia.org/wiki/Truncated_distribution
    # https://en.wikipedia.org/wiki/Exponential_distribution
    # http://lagrange.math.siu.edu/Olive/ch4.pdf
    def __init__(self, L, B=1., random_state=0):
        """
        pdf =
        Args:
            L (float): parameter of the exponential distribution
            B (float): upper bound of the distribution (lower is 0)
            random_state (int): seed to make experiments reproducible
        """
        assert B > 0.
        self.L = L
        self.B = B
        v_m = (1. - np.exp(-B * L) * (1. + B * L)) / L
        super(ArmExp, self).__init__(mean=v_m / (1. - np.exp(-L * B)),
                                     variance=None,  # compute it yourself!
                                     random_state=random_state)

    def cdf(self, x):
        cdf = lambda y: 1. - np.exp(-self.L * y)
        truncated_cdf = (cdf(x) - cdf(0)) / (cdf(self.B) - cdf(0))
        return truncated_cdf

    def inv_cdf(self, q):
        assert 0 <= q <= 1.
        v = - np.log(1. - (1. - np.exp(- self.L * self.B)) * q) / self.L
        return v

    def sample(self):
        # Inverse transform sampling
        # https://en.wikipedia.org/wiki/Inverse_transform_sampling
        q = self.local_random.random_sample(1)
        x = self.inv_cdf(q=q)
        return x


class ArmFinite(AbstractArm):
    def __init__(self, X, P, random_state=0):
        """
        Arm with finite support
        Args:
            X: support of the distribution
            P: associated probabilities
            random_state (int): seed to make experiments reproducible
        """
        self.X = X
        self.P = P
        mean = np.sum(X * P)
        super(ArmFinite, self).__init__(mean=mean,
                                        variance=np.sum(X ** 2 * P) - mean ** 2,
                                        random_state=random_state)

    def sample(self):
        i = self.local_random.choice(len(self.P), size=1, p=self.P)
        reward = self.X[i]
        return reward


class ArmNormal(AbstractArm):

    def __init__(self, mu, sigma, random_state=0):
        self.sigma = sigma
        super(ArmNormal, self).__init__(mean=mu,
                                        variance=sigma ** 2,
                                        random_state=random_state)

    def sample(self):
        x = self.local_random.randn() * self.sigma + self.mean
        return x


if __name__ == '__main__':
    arm = ArmTruncNorm(mean=-1, a=0, b=0.01)
    print(arm.sample())

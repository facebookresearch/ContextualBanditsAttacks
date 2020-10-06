# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import math
from scipy.stats import truncnorm


class ContextualLinearMABModel(object):
    def __init__(self, random_state=0, noise=0.1, theta=None):
        if isinstance(random_state, int):
            self.local_random = np.random.RandomState(random_state)
        else:
            assert isinstance(random_state, np.random.RandomState), "random state is neither an int nor a random number generator"
            self.local_random = random_state
        self.noise = noise
        self.theta = theta

    def reward(self, context, action):
        assert 0 <= action < self.n_actions, "{} not in 0 .. {}".format(action, self.n_actions)
        reward = self.expected_reward(action, context) + self.noise * self.local_random.randn(1)
#        mean = np.dot(self.features[action], self.theta)
#        reward = np.random.binomial(1, mean)
        return reward

    def expected_reward(self, action, context):
        return np.dot(context, self.theta[action])

    def best_expected_reward(self, context):
        D = np.dot(self.theta, context)
        return np.max(D)

    def best_arm(self, context):
        D = np.dot(self.theta, context)
        return np.argmax(D)

    @property
    def n_features(self):
        return self.theta.shape[1]

    @property
    def n_actions(self):
        return self.theta.shape[0]

    def compute_regret(self, context, a_t):
        D = np.dot(self.theta, context)
        return np.max(D) - D[a_t]


class LinContextualArm(object):
    def __init__(self, theta: np.array, random_state:int):
        """
        Args:
            mean: expectation of the arm
            variance: variance of the arm
            random_state (int): seed to make experiments reproducible
        """
        self.theta = theta
        self.local_random = np.random.RandomState(random_state)

    def sample(self, random_state):
        pass


class LinBernoulliArm(LinContextualArm):
    def __init__(self, theta, random_state=0):
        """
        Bernoulli arm
        Args:
             p (float): mean parameter
             random_state (int): seed to make experiments reproducible
        """
        super(LinBernoulliArm, self).__init__(theta=theta, random_state=random_state)

    def sample(self, context: np.array):
        proba = sigmoid(np.dot(self.theta, context))
        return self.local_random.rand(1) < proba


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

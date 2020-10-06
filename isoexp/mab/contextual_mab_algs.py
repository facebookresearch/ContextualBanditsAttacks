# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math

import numpy as np
import sys
import numpy.random as npr
from tqdm import tqdm

class contextEpsGREEDY():
    """
    Args:
        T (int): horizon
        arms (list): list of available arms
    Returns:
        rewards (array-like): observed rewards
        draws (array-like): indexes of selected arms
    """
    def __init__(self, number_arms, dimension, epsilon=0.1, decrease_epsilon=False):

        self.decrease_epsilon = decrease_epsilon
        self.epsilon = epsilon
        self.K = number_arms
        self.dimension = dimension
        self.rewards = []
        self.draws = []

        self.number_pulls = np.ones((self.K,))  # number of observations of each arm
        self.sum_rewards = np.zeros((self.K,))  # sum of rewards for each arm

        self.thetas = np.zeros((self.K, self.dimension))
        self.As = [np.identity(self.dimension) for _ in range(number_arms)]
        self.rewards_matrix = np.zeros((self.K, self.dimension))
        self.nb_iter=0
        self.inv_design_matrices = np.zeros((self.K, dimension, dimension))
        for arm in range(self.K):
            self.inv_design_matrices[arm] = np.eye(dimension, dimension)

    def estimated_best_arm(self, context):
        return np.argmax(self.thetas.dot(context))

    def get_action(self, context, deterministic=False):
        if self.nb_iter < self.K:
            return self.nb_iter
        else:
            # select the chosen_arm
            expected_rewards = self.thetas.dot(context)

            rnd = np.random.rand()
            if not deterministic and rnd <= self.epsilon / (math.sqrt(self.nb_iter + 1) if self.decrease_epsilon else 1):
                chosen_arm = np.random.choice(self.K)
            else:
                idxs = np.flatnonzero(np.isclose(expected_rewards, expected_rewards.max()))
                chosen_arm = np.asscalar(np.random.choice(idxs))
            return chosen_arm

    def update(self, context, chosen_arm, reward):
            # update quantities
            self.nb_iter += 1
            self.rewards.append(reward)
            self.draws.append(chosen_arm)
            self.sum_rewards[chosen_arm] += reward
            self.number_pulls[chosen_arm] += 1

            self.As[chosen_arm] += np.outer(context, context)
            self.rewards_matrix[chosen_arm] += reward * context
            self.thetas[chosen_arm] = np.linalg.inv(self.As[chosen_arm]).dot(self.rewards_matrix[chosen_arm])
            return self.rewards, self.draws


class ContextualLinearBandit(object):
    def __init__(self, nb_arms, dimension, reg_factor=1., delta=0.5,
                 bound_features=None, noise_variance=None, bound_context=None, alpha=None):
        self.K = nb_arms
        self.dim = dimension
        self.reg_factor = reg_factor
        self.delta = delta
        self.exploration_coeff = alpha
        self.iteration = None
        self.bound_context = bound_context
        self.bound_features = bound_features
        self.noise_variance = noise_variance

        self.reset()

    def reset(self):
        d = self.dim
        self.thetas = np.zeros((self.K, d))
        self.inv_design_matrices = np.zeros((self.K, d, d))
        self.bs = self.thetas.copy()
        for arm in range(self.K):
            self.inv_design_matrices[arm] = np.eye(d, d) / self.reg_factor
        # self.range = 1
        # self.est_bound_theta = 0
        # self.est_bound_features = 0
        self.n_samples = np.zeros((self.K,))
        self.iteration = 0

    @property
    def n_actions(self):
        return self.K

    @property
    def n_features(self):
        return self.n_features

    # def auto_alpha(self):
    #     d = self.n_features
    #     sigma, B, D = self.noise_variance, self.bound_theta, self.bound_features
    #     return sigma * np.sqrt(d * np.log((1 + max(1, self.iteration - 1) * D * D / self.reg_factor) / self.delta)) \
    #            + np.sqrt(self.reg_factor) * B

    def alpha(self):
        d = self.dim
        #       print(d)
        sigma, B, D = self.noise_variance, self.bound_context, self.bound_features
        if self.exploration_coeff is None:
            return sigma * np.sqrt(
                d * np.log((1 + np.maximum(1, self.n_samples) * B * B / self.reg_factor) / self.delta)) \
                   + np.sqrt(self.reg_factor) * D
        else:
            return self.exploration_coeff

    def get_action(self, context, deterministic=False):
        self.iteration += 1

        # Let's not be biased with tiebreaks, but add in some random noise
        noise = np.random.random(self.K) * 0.000001
        estimate = np.zeros((self.K,))
        sfactor = self.alpha()
        for arm in range(self.K):
            Ainv = self.inv_design_matrices[arm]
            #             print(Ainv)
            b = self.bs[arm]
            theta_hat = np.dot(Ainv, b)
            self.thetas[arm] = theta_hat
            ta = np.dot(context, np.dot(Ainv, context))
            sfactor = self.alpha()
            # print('sfactor =', sfactor)
            # print('context = ', context)
            # print('theta_hat=', theta_hat)
            # print('ta = ', ta)
            estimate[arm] = np.dot(context, theta_hat) + sfactor[arm] * np.sqrt(ta)
        ucb = estimate + (0 if deterministic else noise)
        choice = np.argmax(ucb)  # choose the highest
        # print(ucb[choice])
        return choice

    def update(self, context, a_t, r_t):

        self.inv_design_matrices[a_t] = self.inv_design_matrices[a_t] - \
                                        np.dot(self.inv_design_matrices[a_t], np.dot(np.outer(context, context),
                                                                                     self.inv_design_matrices[a_t])) \
                                        / (1. + np.dot(context.T, np.dot(self.inv_design_matrices[a_t], context)))
        self.bs[a_t] += r_t * context
        self.n_samples[a_t] += 1

    

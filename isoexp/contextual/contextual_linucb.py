# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import cvxpy as cp
from scipy.optimize import minimize


class RandomArm(object):
    def __init__(self, initial_arms):
        self.arms = initial_arms

    def get_action(self):
        return np.random.choice(self.arms)

    def update(self, a_t, r_t):
        pass

    def reset(self):
        pass

class ContextualLinearBandit(object):
    def __init__(self, nb_arms, dimension, reg_factor=1., delta=0.99,
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
        self.thetas_hat = np.zeros((self.K, d))
        self.inv_design_matrices = np.zeros((self.K, d, d))
        self.bs = self.thetas_hat.copy()
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
        return self.dim

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
            return sigma * np.sqrt(d * np.log((1 + np.maximum(1, self.n_samples) * B * B / self.reg_factor) / self.delta)) \
               + np.sqrt(self.reg_factor) * D
        else:
            return self.exploration_coeff

    def get_action(self, context):
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
            self.thetas_hat[arm] = theta_hat
            ta = np.dot(context, np.dot(Ainv, context))
            sfactor = self.alpha()
            # print('sfactor =', sfactor)
            # print('context = ', context)
            # print('theta_hat=', theta_hat)
            # print('ta = ', ta)
            estimate[arm] = np.dot(context, theta_hat) + sfactor[arm] * np.sqrt(ta)
        ucb = estimate + noise
        choice = np.argmax(ucb)  # choose the highest
        return choice

    def update(self, context, a_t, r_t):

        self.inv_design_matrices[a_t] = self.inv_design_matrices[a_t] - \
                                        np.dot(self.inv_design_matrices[a_t], np.dot(np.outer(context, context),
                                                                                    self.inv_design_matrices[a_t])) \
                                        / (1. + np.dot(context.T, np.dot(self.inv_design_matrices[a_t], context)))
        self.bs[a_t] += r_t * context
        self.n_samples[a_t] += 1
        self.thetas_hat[a_t] = np.dot(self.inv_design_matrices[a_t], self.bs[a_t])


class ContextualLinearTS(object):
    def __init__(self, nb_arms, dimension, reg_factor=1., delta=0.99, noise_variance=None):

        self.K = nb_arms
        self.dim = dimension
        self.delta = delta
        self.reg_factor = reg_factor
        self.noise_variance = noise_variance
        self.reset()

    def reset(self):
        d = self.dim
        self.thetas_hat = np.zeros((self.K, d))
        self.inv_design_matrices = np.zeros((self.K, d, d))
        self.bs = self.thetas_hat.copy()
        for arm in range(self.K):
            self.inv_design_matrices[arm] = np.eye(d, d) / self.reg_factor
        self.n_samples = np.zeros((self.K,))
        self.iteration = 0
        self.thetas = self.thetas_hat

    @property
    def n_actions(self):
        return self.K

    @property
    def n_features(self):
        return self.dim

    def get_action(self, context, deterministic=False):
        self.iteration += 1
        estimate = np.zeros((self.K,))
        nu = self.noise_variance*np.sqrt(self.dim*np.log(self.iteration/self.delta)/2)
        for arm in range(self.K):
            Ainv = self.inv_design_matrices[arm]
            b = self.bs[arm]
            theta_hat = np.dot(Ainv, b)
            self.thetas_hat[arm] = theta_hat
            mean = np.dot(self.thetas_hat[arm], context)
            variance = nu**2 * np.dot(context, np.dot(Ainv, context))
            estimate[arm] = mean + np.sqrt(variance) * (0 if deterministic else np.random.randn())
        ucb = estimate
        choice = np.argmax(ucb)  # choose the highest
        return choice

    def update(self, context, a_t, r_t):

        self.inv_design_matrices[a_t] = self.inv_design_matrices[a_t] - \
                                        np.dot(self.inv_design_matrices[a_t], np.dot(np.outer(context, context),
                                                                                    self.inv_design_matrices[a_t])) \
                                        / (1. + np.dot(context.T, np.dot(self.inv_design_matrices[a_t], context)))
        self.bs[a_t] += r_t * context
        self.n_samples[a_t] += 1


class contextEpsGREEDY():
    """
    Args:
        T (int): horizon
        arms (list): list of available arms
    Returns:
        rewards (array-like): observed rewards
        draws (array-like): indexes of selected arms
    """
    def __init__(self, number_arms, dimension, epsilon=0.1, reg_factor=0.1, decrease_epsilon=False):

        self.decrease_epsilon = decrease_epsilon
        self.epsilon = epsilon
        self.K = number_arms
        self.dim = dimension
        self.rewards = []
        self.draws = []
        self.reg_factor = reg_factor

        self.n_samples = np.ones((self.K,))  # number of observations of each arm
        self.sum_rewards = np.zeros((self.K,))  # sum of rewards for each arm

        self.thetas_hat = np.zeros((self.K, self.dim))
        self.inv_design_matrices = [np.identity(self.dim)/self.reg_factor for _ in range(number_arms)]
        self.bs = np.zeros((self.K, self.dim))
        self.nb_iter = 0
        self.reset()

    def reset(self):
        d = self.dim
        self.thetas_hat = np.zeros((self.K, d))
        self.inv_design_matrices = np.zeros((self.K, d, d))
        self.bs = self.thetas_hat.copy()
        for arm in range(self.K):
            self.inv_design_matrices[arm] = np.eye(d, d) / self.reg_factor
        self.n_samples = np.zeros((self.K,))
        self.nb_iter = 0

    def estimated_best_arm(self, context):
        return np.argmax(self.thetas_hat.dot(context))

    def get_action(self, context):
        if self.nb_iter < self.K:
            return self.nb_iter
        else:
            # select the chosen_arm
            expected_rewards = self.thetas_hat.dot(context)

            rnd = np.random.rand()
            if rnd <= self.epsilon / (np.sqrt(self.nb_iter + 1) if self.decrease_epsilon else 1):
                chosen_arm = np.random.choice(self.K)
            else:
                noise = 10**-7*np.random.randn(self.K)
                chosen_arm = np.argmax(noise + expected_rewards)
            return chosen_arm

    def update(self, context, chosen_arm, reward):
            # update quantities
            self.nb_iter += 1
            self.rewards.append(reward)
            self.draws.append(chosen_arm)
            self.sum_rewards[chosen_arm] += reward
            self.n_samples[chosen_arm] += 1

            self.inv_design_matrices[chosen_arm] = self.inv_design_matrices[chosen_arm] - np.dot(self.inv_design_matrices[chosen_arm], np.dot(np.outer(context, context),
                                                                                    self.inv_design_matrices[chosen_arm])) \
                                        / (1. + np.dot(context, np.dot(self.inv_design_matrices[chosen_arm], context)))
            self.bs[chosen_arm] += reward * context
            self.thetas_hat[chosen_arm] = self.inv_design_matrices[chosen_arm].dot(self.bs[chosen_arm])
            return self.rewards, self.draws


class RewardAttacker(object):
    def __init__(self, nb_arms, dimension, reg_factor=1., delta=0.99,
                 bound_features=None, noise_variance=None, bound_context=None, eps=1/2, **kwargs):
        self.K = nb_arms
        self.dim = dimension
        self.reg_factor = reg_factor
        self.delta = delta
        self.iteration = None
        self.bound_context = bound_context
        self.bound_features = bound_features
        self.noise_variance = noise_variance
        self.eps = eps

        self.reset()

    def reset(self):
        d = self.dim
        self.thetas_hat = np.zeros((self.K, d))
        self.betas = np.zeros((self.K,))
        self.inv_design_matrices = np.zeros((self.K, d, d))
        self.bs = self.thetas_hat.copy()
        for arm in range(self.K):
            self.inv_design_matrices[arm] = np.eye(d, d) / self.reg_factor
        self.n_samples = np.zeros((self.K,))
        self.iteration = 0

    @property
    def n_actions(self):
        return self.K

    @property
    def n_features(self):
        return self.dim

    def alpha(self):
        d = self.dim
        sigma, B, D = self.noise_variance, self.bound_context, self.bound_features
        return sigma * np.sqrt(d * np.log((1 + np.maximum(1, self.n_samples) * B * B / self.reg_factor) / self.delta)) \
               + np.sqrt(self.reg_factor) * D

    def update(self, context, a_t, r_t):
        self.inv_design_matrices[a_t] = self.inv_design_matrices[a_t] - \
                                        np.dot(self.inv_design_matrices[a_t], np.dot(np.outer(context, context),
                                                                                     self.inv_design_matrices[a_t])) \
                                        / (1. + np.dot(context.T, np.dot(self.inv_design_matrices[a_t], context)))
        self.bs[a_t] += r_t * context
        self.n_samples[a_t] += 1
        self.thetas_hat[a_t] = np.dot(self.inv_design_matrices[a_t], self.bs[a_t])
        self.betas = self.alpha()

    def compute_attack(self, action, context, a_star):
        if action != a_star:
            temp_1 = self.betas[action] * np.sqrt(np.dot(context, np.dot(self.inv_design_matrices[action], context)))
            temp_2 = self.betas[a_star] * np.sqrt(np.dot(context, np.dot(self.inv_design_matrices[a_star], context)))
            att = - min(1, max(np.dot(self.thetas_hat[action], context) + temp_1, 0)) + (1 - self.eps) \
                  * (min(1, max(0, np.dot(self.thetas_hat[a_star], context) - temp_2)))
            return att
        else:
            return 0

class Exp4(object):
    def __init__(self, nb_arms, dimension, experts,  eta=0.5, gamma=1):
        self.K = nb_arms
        self.dim = dimension
        self.eta = eta
        self.gamma = gamma
        self.experts = experts
        self.nb_experts = len(experts)
        self.reset()

    @property
    def n_actions(self):
        return self.K

    @property
    def n_features(self):
        return self.dim

    def reset(self):
        self.Q = np.ones((self.nb_experts,))/self.nb_experts
        self.iteration = 0

    def get_expert_advice(self, context):
        proba_matrix = np.zeros((self.nb_experts, self.K))
        for m in range(self.nb_experts):
            proba_matrix[m] = self.experts[m].get_action(context)
        return proba_matrix

    def get_action(self, context):
        self.iteration += 1
        self.E = self.get_expert_advice(context)
        self.P = np.dot(self.E.T, self.Q)
        #self.P = self.P/np.sum(self.P)
        temp = np.linspace(0, self.K-1, self.K, dtype='int')
        action = np.random.choice(temp, p=self.P)
        return action

    def update(self, context, a_t, r_t):
        X = np.ones((self.K,))
        X[a_t] = X[a_t] - (1 - r_t)/(self.P[a_t] + self.gamma)
        X_experts = np.dot(self.E, X)
        self.Q = self.Q*np.exp(self.eta*X_experts)
        self.Q = self.Q/np.sum(self.Q)

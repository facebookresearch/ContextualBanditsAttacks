# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import pickle
import matplotlib.pyplot as plt


class ContextualLinearMABModel(object):
    def __init__(self, random_state=0, noise=0.1, thetas=None):
        self.local_random = np.random.RandomState(random_state)
        self.noise = noise
        self.thetas = thetas

    def reward(self, context, action):
        assert 0 <= action < self.n_actions, "{} not in 0 .. {}".format(action, self.n_actions)
        reward = np.dot(context, self.thetas[action]) + self.noise * self.local_random.randn(1)
        return reward

    def best_arm_reward(self, context):
        D = np.dot(self.thetas, context)
        return np.max(D)

    @property
    def n_features(self):
        return self.thetas.shape[1]

    @property
    def n_actions(self):
        return self.thetas.shape[0]

class RandomContextualLinearArms(ContextualLinearMABModel):
    def __init__(self, random_state=0, noise=0., n_actions=4, n_features=4, bound_context=1):
        self.bound_context = bound_context
        thetas = np.abs(np.random.randn(n_actions, n_features-1))
        super(RandomContextualLinearArms, self).__init__(random_state=random_state, noise=noise,
                                                         thetas=thetas)
        self.context_lists = []
        self.n_user = 5
        self.n = self.n_user
        thetas = np.ones((n_actions, n_features))
        thetas[:, :-1] = self.thetas.copy()
        max_rwd = -float('inf')
        min_rwd = float('inf')
        for k in range(self.n_user):
            test = np.abs(np.random.randn(self.n_features))
            test = np.random.uniform(low=0, high=bound_context)*test/np.linalg.norm(test)
            dot_prod = np.dot(self.thetas, test)
            maxi = np.max(dot_prod)
            mini = np.min(dot_prod)
            if maxi >= max_rwd:
                max_rwd = maxi
            if mini <= min_rwd:
                min_rwd = mini
            self.context_lists.append(np.concatenate((test, np.array([1]))))
        self.thetas = thetas
        thetas[:, -1] = -min_rwd + 1
        thetas = thetas / (max_rwd - min_rwd + 1)
        self.thetas = thetas
        print('Different Means:')
        for k in range(self.n_user):
            print('Means for context {}'.format(k), np.dot(thetas, self.context_lists[k]))
        self.theta = self.thetas

    def get_context(self):
        return self.context_lists[np.random.randint(low=0, high=self.n_user)]


class DatasetModel(ContextualLinearMABModel):
    def __init__(self, arm_csvfile, user_csvfile, random_state=0, noise=0., arms_limit=None, context_limit=None):
        temp_thetas = np.loadtxt(arm_csvfile, delimiter=',').T
        temp_user_contexts = np.loadtxt(user_csvfile, delimiter=',')
        K, d = temp_thetas.shape
        N, _ = temp_user_contexts.shape
        thetas = np.ones((K, d+1))
        user_contexts = np.ones((N, d+1))
        thetas[:, :-1] = temp_thetas.copy()
        if arms_limit is not None:
            thetas = thetas[:arms_limit]
        user_contexts[:, :-1] = temp_user_contexts.copy()
        if context_limit is not None:
            user_contexts = user_contexts[:context_limit]
        self.bound_context = np.linalg.norm(user_contexts, axis=1).max()
        D = np.dot(temp_user_contexts, temp_thetas.T)
        min_rwd = np.min(D)
        max_rwd = np.max(D)
        thetas[:, -1] = -min_rwd + 1
        thetas = thetas / (max_rwd - min_rwd + 1)
        self.context_lists = user_contexts.copy()
        self.n_user, _ = user_contexts.shape
        super(DatasetModel, self).__init__(random_state=random_state, noise=noise,
                                           thetas=thetas)
        self.theta = self.thetas

    def get_context(self):
        return self.context_lists[np.random.randint(low=0, high=self.n_user)]





class AttackOneUserModel(ContextualLinearMABModel):
    def __init__(self, random_state=0, noise=0., n_actions=4, n_features=4, bound_context=1, distance=1):
        self.bound_context = bound_context
        thetas = np.abs(np.random.randn(n_actions, n_features))
        norm_thetas = np.linalg.norm(thetas, axis=1)
        thetas = (1/2) * thetas/norm_thetas.reshape((n_actions, 1))
        super(AttackOneUserModel, self).__init__(random_state=random_state, noise=noise,
                                                         thetas=thetas)
        self.context_lists = []
        self.n_user = 1
        self.n = self.n_user
        self.distance = distance
        for k in range(self.n_user):
            test = np.abs(np.random.randn(self.n_features))
            test = np.random.uniform(low=0, high=bound_context)*test/np.linalg.norm(test)
            self.context_lists.append(test)
        print('Different Means:')
        # for k in range(self.n_user):
        #     print('Means for context {}'.format(k), np.dot(thetas, self.context_lists[k]))
        self.theta = self.thetas

    def get_context(self):
        return self.context_lists[np.random.randint(low=0, high=self.n_user)]

    def add_target_arm(self):
        theta_target_arm = np.abs(np.random.randn(self.n_features))
        theta_target_arm = self.distance * theta_target_arm/np.linalg.norm(theta_target_arm)
        import cvxpy as cp
        n_points = len(self.thetas)
        lambdas = cp.Variable(n_points)
        A = np.ones((1,n_points))
        pos = -np.eye(n_points)
        constraints = [A@lambdas == 1, pos@lambdas <= 0]
        obj = cp.Minimize(cp.quad_form(theta_target_arm - self.thetas.T @ lambdas, np.eye(self.n_features)))
        prob = cp.Problem(obj, constraints)
        prob.solve()
        print('Distance to convex hull', np.sqrt(prob.value))
        self.thetas = np.concatenate((self.thetas, theta_target_arm.reshape((1, self.n_features))), axis=0)

if __name__ == '__main__':
    import os
    # arm_file = os.path.join(os.getcwd(),'../../examples/jester/Vt_jester.csv')
    # user_file = os.path.join(os.getcwd(),'../../examples/jester/U.csv')
    # test_model = DatasetModel(arm_csvfile=arm_file, user_csvfile=user_file, context_limit=100)
    r = np.linspace(0, 1/2)
    for rr in r:
        test_model = AttackOneUserModel(n_features=2, n_actions=10, distance=rr)
        # print(test_model.context_lists)
        # print(np.linalg.norm(test_model.thetas,axis=1))
        test_model.add_target_arm()
    # print(test_model.thetas)
    # for x in test_model.context_lists:
    #     print(np.dot(test_model.thetas, x))
    if test_model.n_features == 2:
        for a in range(test_model.n_actions-1):
            plt.scatter(test_model.thetas[a, 0], test_model.thetas[a, 1], marker='+')
        plt.scatter(test_model.thetas[test_model.n_actions - 1, 0], test_model.thetas[test_model.n_actions - 1, 1],
                    marker='^')
        # for x in test_model.context_lists:
        #     plt.scatter(x[0], x[1], marker='o')
    plt.show()
    # RandomContextualLinearArms()
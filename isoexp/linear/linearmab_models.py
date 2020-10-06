# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np


class LinearMABModel(object):
    def __init__(self, random_state=0, noise=0.1, features=None, theta=None):
        self.local_random = np.random.RandomState(random_state)
        self.noise = noise
        self.features = features
        self.theta = theta

    def reward(self, action):
        assert 0 <= action < self.n_actions, "{} not in 0 .. {}".format(action, self.n_actions)
        reward = np.dot(self.features[action], self.theta) + self.noise * self.local_random.randn(1)
#        mean = np.dot(self.features[action], self.theta)
#        reward = np.random.binomial(1, mean)

        return reward

    def best_arm_reward(self):
        D = np.dot(self.features, self.theta)
        return np.max(D)

    @property
    def n_features(self):
        return self.features.shape[1]

    @property
    def n_actions(self):
        return self.features.shape[0]


class LinPHEModel(object):

    def __init__(self, d=10, n_actions=100, random_state=0):
        self.local_random = np.random.RandomState(random_state)
        self.n_features = d
        self.n_actions = n_actions
        temp_theta = self.local_random.randn(d - 1)
        temp_theta = np.random.uniform(0, 1 / 2) * temp_theta / np.linalg.norm(temp_theta)
        self.theta = np.ones(d) / 2
        self.theta[:-1] = temp_theta

        self.features = np.ones((n_actions, d))
        # temp_features = self.local_random.randn(n_actions, d-1)
        # temp_features = np.random.uniform(0, 1)*temp_features/np.linalg.norm(temp_features, axis = 1).reshape((self.n_actions, 1))
        # print(temp_features)
        # self.features[:, :-1] = temp_features

        radius = 1
        Y = self.local_random.randn(n_actions, d - 1)
        U = np.random.uniform(0, 1, size=n_actions)
        r = radius * np.power(U, 1. / (d - 1))
        F = r / np.linalg.norm(Y, axis=1)
        X = Y * F[:, np.newaxis]
        self.features[:, :-1] = X

        # import matplotlib.pyplot as plt
        # from mpl_toolkits.mplot3d import Axes3D
        # if d-1 == 3:
        #     fig = plt.figure()
        #     ax = fig.add_subplot(111, projection='3d')
        #     ax.scatter(X[:,0], X[:,1], X[:,2], label='new')
        #     ax.scatter(temp_features[:,0], temp_features[:,1], temp_features[:,2], label='old')
        #     plt.legend()
        #     plt.show()
        # if d-1 == 2:
        #     plt.figure()
        #     plt.scatter(X[:,0], X[:,1], label='new')
        #     plt.scatter(temp_features[:,0], temp_features[:,1], label='old')
        #     plt.legend()
        #     plt.show()
        #
        # print(X)

    def reward(self, action):
        assert 0 <= action < self.n_actions, "{} not in 0 .. {}".format(action, self.n_actions)
        reward = self.local_random.binomial(1, np.dot(self.theta, self.features[action]))
        return reward

    def best_arm_reward(self):
        D = np.dot(self.features, self.theta)
        return np.max(D)

    def means(self):
        D = np.dot(self.features, self.theta)
        return D

class RandomLogArms(object) :
    def __init__(self, random_state = 0, noise = .1, 
                 n_actions = 4, n_features = 100, 
                 bound_features = 1, bound_theta = 1) :
        
        features = np.random.randn(n_actions, n_features)
        self.features = bound_features*features/max(np.linalg.norm(features, axis = 1))
        theta = np.random.randn(n_features)
        self.theta =  np.random.uniform(low = 0, high = bound_theta)*theta/np.linalg.norm(theta)
        self.link = lambda x : 1/(1 + np.exp(-x))
        self.noise = noise
        self.local_random = np.random.RandomState(random_state)
        self.n_actions, self.n_features = n_actions, n_features
        temp = np.dot(self.features,self.theta) + bound_features
        self.kappa = min(self.link(temp)*(1 - self.link(temp)))
    def reward(self, action) :
        reward = self.link(np.dot(self.features[action], self.theta)) + self.noise * self.local_random.randn(1)
        return reward
    
    def best_arm_reward(self):
        D = np.dot(self.features, self.theta)
        return self.link(np.max(D))


class RandomNormalLinearArms(LinearMABModel):
    def __init__(self, random_state=0, noise=0., n_actions=4, n_features=4, reward_lim=(-np.inf, np.inf)):
        features = np.random.randn(n_actions, n_features)
        real_theta = np.random.randn(n_features) * 0.5

        means = np.dot(features, real_theta)
        idxs = (means < reward_lim[0]) | (means > reward_lim[1])
        idxs = np.arange(n_actions)[idxs]
        for i in idxs:
            mean = -np.inf
            feat = None
            while mean > reward_lim[1] or mean < reward_lim[0]:
                feat = np.random.randn(1, n_features)
                mean = np.dot(feat, real_theta)
            features[i, :] = feat

        super(RandomNormalLinearArms, self).__init__(random_state=random_state, noise=noise,
                                                     features=features, theta=real_theta)


class RandomLinearArms(LinearMABModel):
    def __init__(self, random_state=0, noise=0., n_actions=4, n_features=4, bound_features=1, bound_theta = 1, positive=True, max_one=True):
        features = np.random.randn(n_actions, n_features)
        real_theta = np.random.randn(n_features)
        real_theta = np.random.uniform(low = 1/2, high = bound_theta)*real_theta/np.linalg.norm(real_theta)
        if positive:
            idxs = np.dot(features, real_theta) <= 0
            idxs = np.arange(n_actions)[idxs]
            for i in idxs:
                mean = -1
                feat = None
                while mean <= 0:
                    feat = np.random.randn(1, n_features)
                    mean = np.dot(feat, real_theta)
                features[i, :] = feat
        features = np.random.uniform(low = 1/2, high = bound_features, size = (n_actions,1)) * features / max(np.linalg.norm(features, axis=1))

        if max_one:
            D = np.dot(features, real_theta)

            min_rwd = min(D)
            max_rwd = max(D)
            min_features = features[np.argmin(D)]
            features = (features - min_features) / (max_rwd - min_rwd)

        super(RandomLinearArms, self).__init__(random_state=random_state, noise=noise,
                                               features=features, theta=real_theta)


class DiffLinearArms(LinearMABModel):
    def __init__(self, random_state=0, noise=0., n_actions=4, n_features=2, real_theta=np.array([9 / 10, 1 / 10]),
                 optimal_arm=np.array([1, 0]), baseline_arm=np.array([0, 1]), concentration_coeff=0.3):
        baseline_arm = baseline_arm.reshape((baseline_arm.shape[0], 1))
        features = baseline_arm + concentration_coeff * np.random.randn(n_features, n_actions)
        idxs = np.dot(real_theta, features) <= 0
        idxs = np.arange(n_actions)[idxs]
        for i in idxs:
            mean = -1
            feat = None
            while mean <= 0:
                feat = baseline_arm + concentration_coeff * np.random.randn(n_features, 1)
                mean = float(np.dot(real_theta, feat))
            features[:, i] = feat.squeeze()
        optimal_arm = optimal_arm.reshape((optimal_arm.shape[0], 1))
        features = np.concatenate((features, optimal_arm), axis=1)
        features = np.concatenate((features, baseline_arm), axis=1)

        super(DiffLinearArms, self).__init__(random_state=random_state, noise=noise, features=features,
                                             theta=real_theta)


class OtherArms(LinearMABModel):
    def __init__(self, random_state=0, noise=0., n_actions=4, n_features=2):
        angular_fraction = np.linspace(0, np.pi / 2, n_actions)
        features = np.zeros((n_actions, n_features))
        features[:, 0] = np.cos(angular_fraction)
        features[:, 1] = np.sin(angular_fraction)
        real_theta = np.array([1 - np.pi / (4 * n_actions), np.pi / (4 * n_actions)])
        super(OtherArms, self).__init__(random_state=random_state, noise=noise, features=features, theta=real_theta)


class CircleBaseline(LinearMABModel):
    def __init__(self, random_state=0, noise=0., n_actions=4, n_features=2, inner_radius=1 / 10, outer_radius=2):
        temp = np.random.uniform(0, 2 * np.pi)
        theta = outer_radius * np.array([np.cos(temp), np.sin(temp)])

        angle_baseline = np.random.uniform(0, 2 * np.pi)
        radius_baseline = np.random.uniform(1 / 10, inner_radius)
        baseline = radius_baseline * np.array([np.cos(angle_baseline), np.sin(angle_baseline)]).reshape(1, n_features)
        features = np.zeros((n_actions - 1, n_features))
        radius_features = np.random.uniform(low=2 * inner_radius, high=outer_radius, size=(n_actions - 1, 1))
        # radius_features = np.random.uniform(low = 0, high = inner_radius, size = (n_actions-1,1))
        angle_features = np.random.uniform(0, 2 * np.pi, size=n_actions - 1)
        features[:, 0] = np.cos(angle_features)
        features[:, 1] = np.sin(angle_features)
        features = radius_features * features
        features = np.concatenate((features, baseline), axis=0)
        # features = np.concatenate((baseline, features), axis = 0)
        super(CircleBaseline, self).__init__(random_state=random_state, noise=noise, features=features, theta=theta)

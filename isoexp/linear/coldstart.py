# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
from .linearmab_models import LinearMABModel


class ColdStartFromDatasetModel(LinearMABModel):
    def __init__(self, arm_csvfile, user_csvfile, random_state=0, noise=0.):
        features = np.loadtxt(arm_csvfile, delimiter=',').T
        thetas = np.loadtxt(user_csvfile, delimiter=',')

        super(ColdStartFromDatasetModel, self).__init__(random_state=random_state, noise=noise,
                                                        features=features, theta=None)

        self.theta_idx = np.random.randint(low=0, high=thetas.shape[0])
        print("Selecting user: {}".format(self.theta_idx))
        self.theta = thetas[self.theta_idx]
        # self.theta = np.random.randn(thetas.shape[1])

        D = np.dot(self.features, self.theta)

        min_rwd = min(D)
        max_rwd = max(D)
        min_features = features[np.argmin(D)]
        self.features = (self.features - min_features) / (max_rwd - min_rwd)


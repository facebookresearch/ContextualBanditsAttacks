# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from isoexp.isotonicsim import LIsotron
import matplotlib.pyplot as plt
import numpy as np
from isoexp.LPAV_cvx import cvx_lip_isotonic_regression


N = 500
m = 5
X = np.random.rand(N*m).reshape(N, m)
w = np.random.rand(m)
orda = np.argsort(np.dot(X, w))
X = X[orda, :]
y = 2*np.dot(X, w)
y = np.dot(X, w)**3 # + np.random.randn(N)
x = np.dot(X, w)

#reg = LIsotron()
#yn = reg.fit_transform(X, y, lipschitz_value=1, iterations=50)

ones = np.zeros_like(y)
iterations=400
wt = np.random.rand(X.shape[1])
wt = np.zeros(X.shape[1])

for t in range(iterations):
    zt = np.dot(X, wt)
    order = np.argsort(zt)
    zt = zt[order]
    print(zt)
    y_iso = cvx_lip_isotonic_regression(x=zt, y=y[order], weights=ones, lipschitz_value=10)
    print(y_iso)
    print(y)
    # plt.plot(xt, y[order], 'o')
    # plt.plot(xt, y_iso, '+')
    # plt.show()
    wt = wt + np.mean((y[order] - y_iso)[:, np.newaxis] * X[order, :], axis=0)

print("true weights: {}".format(w))
print("weights:      {}".format(wt))
plt.figure()
plt.plot(np.dot(X, w), y, '+', label="true")
#plt.plot(np.dot(X, w), np.dot(X, wt), 'o', label="learnt")
plt.plot(np.dot(X, w), y_iso, 'o', label="learnt2")
plt.legend()
plt.show()


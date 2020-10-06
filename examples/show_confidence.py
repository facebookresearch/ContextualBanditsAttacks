# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import numpy as np
from isoexp.monenvs import Env1
from sklearn.isotonic import IsotonicRegression
import matplotlib.pyplot as plt
import math

np.random.seed(521524)

from sklearn.utils import check_random_state
from isoexp._samplers import isotonic_data_bands


def paper_f(X, sigma):
    v = 20 * X / 0.4 - (10 + 0.3 * 20 / 0.4)
    v[X <= 0.3] = -10
    v[X > 0.7] = 10
    return v + sigma * np.random.randn(len(X))


env = Env1()
N = 450
sigma = 0.5
X = np.random.rand(N)
X = np.sort(X)
Y = env.f[0](X) + sigma * np.random.randn(N)
Y = paper_f(X, sigma=sigma)
# Y = (10*X)**3/ 3.4 + sigma * np.random.randn(N)

# X = np.random.rand(N)
# X = np.sort(X)
# rs = check_random_state(312312)
# Y = rs.randint(-10, 10, size=(N,)) + 10. * np.log1p(np.arange(N))

L = 20/0.4
# L = 3 * 100

plt.plot(X, Y, 'o', label="Y")

idx_vector = np.arange(N)

ir = IsotonicRegression()
ir = ir.fit(X, Y)
Y_iso = ir.transform(X)
plt.plot(X, Y_iso, '-d', label="iso(Y)")
plt.legend()

T = np.linspace(0.001, 0.999, 50)
f = ir.predict(T)
f[T < X[0]] = Y_iso[0]
f[T > X[-1]] = Y_iso[-1]

delta = 0.1

# for idx in range(len(T)):
#     X_new = T[idx]
#     if X_new < X[0]:
#         lb = -L * np.abs(X_new - X[0]) - np.sqrt(2*np.log((N**2 + N)/delta))
#         lbm = 1
#         m = 1
#         ub = np.inf
#         while m <= N:
#             val = np.mean(Y_iso[0:m]) - f[idx] + np.sqrt(2*np.log((N**2 + N)/delta) / m)
#             if val < ub:
#                 ub = val
#                 ubm = m
#             m += 1
#     elif X_new > X[-1]:
#         ub = L * np.abs(X_new - X[-1]) + np.sqrt(np.log(2*(N**2 + N)/delta))
#         ubm = 1
#         m = 1
#         while m <= N:
#             val = np.mean(Y_iso[N-m:N]) - f[idx] - np.sqrt(np.log(2*(N**2 + N)/delta) / m)
#             if val > lb:
#                 lb = val
#                 lbm = m
#             m += 1
#     else:
#
#         k = np.max(idx_vector[(X_new > X)]) + 1
#         assert k == (np.sum(X_new > X)), "{},{}".format(k, (np.sum(X_new > X)))
#         m = 1
#         mtop = max(k, N - k)
#         ub = np.inf
#         lb = -np.inf
#         while m <= mtop:
#             if m <= k:
#                 val = np.mean(Y_iso[k - m:k + 1]) - f[idx] - np.sqrt(np.log(2*(N**2 + N)/delta) / m)
#                 if val > lb:
#                     lb = val
#                     lbm = m
#
#             if m <= N - k:
#                 val = np.mean(Y_iso[k:k + m]) - f[idx] + np.sqrt(np.log(2*(N**2 + N)/delta) / m)
#
#                 if val < ub:
#                     ub = val
#                     ubm = m
#             m += 1
#
#     print(X_new, lbm, lb, ub, ubm)
#     plt.plot(X_new, f[idx] + ub, 'r+')
#     plt.plot(X_new, f[idx] + lb, 'g*')


# idx = N - 7
# print("N: {}".format(N))
# print(T[idx], f[idx])
# lb, ub = isotonic_data_bands(X, Y, T[idx], f[idx], L, sigma, delta)
# print()
# print(lb, ub)
# exit(312)

LUCB = np.zeros((len(T), 4))

plt.figure()
plt.plot(T, f, ':+', label="iso(t)")
plt.plot(X, Y_iso, 'o', label="iso(Y)")
for idx in range(len(T)):
    X_new = T[idx]
    y_new = f[idx]

    lb, ub = isotonic_data_bands(X, Y_iso, X_new, y_new, L, sigma, delta)
    LUCB[idx, 0:2] = [lb, ub]
    # plt.plot(X_new, y_new + ub, 'r+', label="ub-iso(Y)")
    # plt.plot(X_new, y_new + lb, 'g*', label="lb-iso(Y)")

# plt.figure()
# plt.plot(T, f, '+')
for idx in range(len(T)):
    X_new = T[idx]
    y_new = f[idx]

    lb, ub = isotonic_data_bands(X, Y, X_new, y_new, L, sigma, delta)
    LUCB[idx, 2:4] = [lb, ub]
    # plt.plot(X_new, y_new + ub, 'r*', label="ub-Y")
    # plt.plot(X_new, y_new + lb, 'g-', label="lb-Y")


print(LUCB)
plt.plot(T, f + LUCB[:,0], 'g*', label="lb-iso(Y)")
plt.plot(T, f + LUCB[:,1], 'r+', label="ub-iso(Y)")
plt.plot(T, f + LUCB[:,2], 'b4', label="lb-Y")
plt.plot(T, f + LUCB[:,3], 'ko', label="ub-Y", fillstyle='none')
plt.legend()
plt.show()


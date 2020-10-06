# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from isoexp.knnmab import KnnMab
from isoexp.isomab import IsoMab
import isoexp.monenvs as monenvs
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time

np.random.seed(12345)

env = monenvs.Env1()
env.show()
plt.show()

# define algorithms
knn_a = KnnMab(env=env, Lc=100)
iso_a = IsoMab(env=env, Lc=100)
algs = [(knn_a, "knn"),(iso_a, "iso")]

# define params
rep = 2
T = 2500

## force compilation of the function
from isoexp.knnmab import knn_select_arm
start = time.time()
knn_select_arm(np.zeros((4,5)), np.zeros((4,)), -1, 1, 3, 1)
end = time.time()
print("Elapsed (with compilation) = %s" % (end - start))

# from isoexp.isomab import construct_datadep_ci
# start = time.time()
# construct_datadep_ci(np.zeros((6,)), np.zeros((6,)), np.zeros((1,)), np.zeros((1,)), 1, -1)
# end = time.time()
# print("Elapsed (with compilation) = %s" % (end - start))

# prepare figure
rf = plt.figure(99)

for alg in algs:
    regrets = np.zeros((rep, T))
    for i in tqdm(range(rep)):
        alg[0].reset()
        regret = alg[0].run(iterations=T)
        cr = np.cumsum(regret)
        regrets[i,:] = cr

    plt.figure(99)
    mu = regrets.mean(axis=0)
    sigma = regrets.std(axis=0) / np.sqrt(rep)
    p = plt.plot(mu, label=alg[1])
    plt.fill_between(np.arange(len(mu)), mu + 2*sigma, mu - 2*sigma, facecolor=p[-1].get_color(), alpha=0.5)

    plt.figure()
    X = np.linspace(0, 1, 100)
    arms = np.zeros_like(X)
    for i in range(len(X)):
        arms[i] = alg[0].select_arm(np.array([X[i]]))
    plt.plot(X, arms, '+')
    plt.title("Arm selection")
    plt.xlabel("Covariate X")
    plt.ylabel("Arm")
    plt.title(alg[1])

plt.figure(99)
plt.xlabel("Time")
plt.ylabel("Cumulative Regret")
plt.legend()
plt.show()

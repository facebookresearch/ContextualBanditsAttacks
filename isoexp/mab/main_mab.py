# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import sys
sys.path.append('/isoexp')
import numpy as np
import isoexp.mab.arms as arms
import pickle
from isoexp.mab.smab_algs import UCB1, EXP3_IX, attacked_UCB1, attacked_EXP3_IX, EXP3_P, attacked_EXP3_P, FTRL, attacked_FTRL
from matplotlib import rc
import json
import datetime

rc('text', usetex=True)

import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import namedtuple

MABResults = namedtuple('MABResults', 'regret, cum_rewards, attacks, times_of_attacks')

random_state = np.random.randint(0, 123123)

K = 5
MAB = []
means = np.random.uniform(low=0.25, high=0.75, size=K)
#means = np.array([0.47823152, 0.70243227, 0.64504063, 0.65679234, 0.49546542,
#       0.46417188, 0.64736977, 0.71255566, 0.66844984, 0.26030838])
for k in range(K) :
    #MAB.append(arms.ArmBeta(a=8*means[k], b=8*(1-means[k])))
    MAB.append(arms.ArmBernoulli(p=means[k]))
nb_arms = len(MAB)
print('means: {}'.format(means))
mu_max = np.max(means)
a_star = np.argmin(means)
T = 1*10**4# horizon
nb_simu = 10
eta = np.sqrt(2*np.log(K + 1)/(K*T))
# eta = 0.01
gamma = eta/2
settings = {
    "T": T,
    "nb_simu": nb_simu,
    "random_state": random_state,
    "K": K,
}



algorithms = {
    #'EXP3': lambda T, MAB: FTRL(T, MAB, eta=eta, alg='epx_3'),
    'INF': lambda T, MAB: FTRL(T, MAB, eta=eta, alg='inf'),
    'Attacked INF': lambda T, MAB: attacked_FTRL(T, MAB, target_arm=a_star, eta=eta, alg='inf'),
    # 'FTRL log barrier' : lambda T, MAB: FTRL(T, MAB, eta=eta, alg='log_barrier'),
    # 'Attacked FTRL log barrier': lambda T, MAB: attacked_FTRL(T, MAB, target_arm=a_star, eta=eta, alg='log_barrier'),
    'UCB': lambda T, MAB: UCB1(T, MAB, alpha=1),
    'Attacked UCB': lambda T, MAB: attacked_UCB1(T, MAB, target_arm = a_star, alpha=1., delta=0.99),
    # 'EXP3-IX': lambda T, MAB: EXP3_IX(T, MAB, eta=eta, gamma=gamma),
    # 'Attacked EXP3-IX': lambda T, MAB: attacked_EXP3_IX(T, MAB, target_arm=a_star),
     'EXP3': lambda T, MAB: EXP3_P(T, MAB, eta=np.sqrt(np.log(K)/(T*K))),
      'Attacked EXP3': lambda T, MAB: attacked_EXP3_P(T, MAB, target_arm=a_star),
    # 'EXP3.P Gamma 0.1': lambda T, MAB: EXP3_P(T, MAB, gamma=0.1, eta=np.sqrt(np.log(K)/(K*T))),
    # 'Attacked EXP3.P Gamma 0.1': lambda T, MAB: attacked_EXP3_P(T, MAB, target_arm=a_star, gamma=0.1, eta=np.sqrt(np.log(K)/(K*T)))
}
results = []
full_algo = algorithms

for alg_name in full_algo.keys():

    alg = full_algo[alg_name]

    regret = np.zeros((nb_simu, T))
    rwds = 0*regret
    times = 0*regret
    attacks = 0*regret

    for k in tqdm(range(nb_simu), desc='Simulating {}'.format(alg_name)):
        try:
            rewards, draws = alg(T, MAB)
        except ValueError:
            rewards, draws, att, times_of_att = alg(T, MAB)
            attacks[k] = np.cumsum(att)
            times[k] = times_of_att
        rwds[k] = np.cumsum(means[draws.astype('int')])
        regret[k] = max(means) * np.arange(1, T + 1) - rwds[k]

    results += [(alg_name, MABResults(regret=regret, cum_rewards=rwds, attacks=attacks, times_of_attacks=times))]

id = '{:%Y%m%d_%H%M%S}_{}'.format(datetime.datetime.now(), 'GLM')
with open("{}_{}_MAB_illustration.pickle".format(id, "SEQ"), "wb") as f:
    pickle.dump(results, f)
with open("{}_{}_MAB_illustration_settings.json".format(id, "SEQ"), "w+") as f:
    json.dump(settings, f)


t = np.arange(0, T)
for alg_name, val in results:
    mean_regret = np.mean(val.regret, axis=0)
    low_quantile_regret = np.quantile(val.regret, 0.25, axis=0)
    high_quantile_regret = np.quantile(val.regret, 0.75, axis=0)
    rwds = np.mean(val.cum_rewards, axis=0)
    low_quantile_rwds = np.quantile(val.cum_rewards, 0.25, axis=0)
    high_quantile_rwds = np.quantile(val.cum_rewards, 0.75, axis=0)
    plt.figure(1)
    plt.title('Rewards')
    plt.plot(rwds, label=alg_name)
    plt.legend()
    plt.fill_between(t, low_quantile_rwds, high_quantile_rwds, alpha=0.15)
    plt.figure(2)
    plt.title('Regret')
    plt.plot(mean_regret, label=alg_name)
    plt.legend()
    plt.fill_between(t, low_quantile_regret, high_quantile_regret, alpha=0.15)
    if 'Attacked' in alg_name:
        plt.figure(3)
        cum_sum_attacks = np.mean(np.abs(val.attacks), axis=0)
        low_quantile_attacks = np.quantile(np.abs(val.attacks), 0.25, axis=0)
        high_quantile_attacks = np.quantile(np.abs(val.attacks), 0.75, axis=0)
        plt.title('Cumulative sum of attacks')
        plt.plot(cum_sum_attacks, label=alg_name)
        plt.legend()
        plt.fill_between(t, low_quantile_attacks, high_quantile_attacks, alpha=0.15)
        # plt.figure(2)
        # rep = np.random.randint(low=0, high=nb_simu)
        # times_to_consider = val.times_of_attacks[rep]
        # plt.scatter(t[times_to_consider == 1], val.regret[rep, times_to_consider == 1])
        plt.figure(4)
        plt.title('Number of attacks')
        number_of_attacks = np.mean(np.cumsum(val.times_of_attacks, axis=1), axis=0)
        high_quantile = np.quantile(np.cumsum(val.times_of_attacks, axis=1), 0.75, axis=0)
        low_quantile = np.quantile(np.cumsum(val.times_of_attacks, axis=1), 0.25, axis=0)
        plt.plot(number_of_attacks, label=alg_name)
        plt.legend()
        plt.fill_between(t, low_quantile, high_quantile, alpha=0.15)
plt.show()

#import tikzplotlib
#tikzplotlib.save("lcb_worst.tex")

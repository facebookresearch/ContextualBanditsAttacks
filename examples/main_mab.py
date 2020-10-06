# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import isoexp.mab.arms as arms
import pickle
from isoexp.mab.smab_algs import UCB1, UCBV, BootstrapedUCB, PHE, Random_exploration
from isoexp.conservative.mab import CUCBV, SafetySetUCBV, powerful_oracle, CBUCB, CUCB, CPHE
from matplotlib import rc
import json
import datetime

rc('text', usetex=True)

import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import namedtuple

MABResults = namedtuple('MABResults', 'regret, cum_rewards')

random_state = np.random.randint(0, 123123)
random_state = 117060

K = 10
MAB = []
means = np.random.uniform(low = 0.25, high = 0.75, size = K)
means = np.array([0.47823152, 0.70243227, 0.64504063, 0.65679234, 0.49546542,
       0.46417188, 0.64736977, 0.71255566, 0.66844984, 0.26030838])
for k in range(K) :
    MAB.append(arms.ArmBernoulli(p = means[k]))
nb_arms = len(MAB)
print('means: {}'.format(means))
mu_max = np.max(means)

T = 10000# horizon
nb_simu = int(np.sqrt(T))

#Define baseline
pos = 3
baseline = np.argsort(means)[pos]
mean_baseline = MAB[baseline].mean

#Batch Version 
conservative_level = 0.1
check_every = 2*np.sqrt(T)

settings = {
    "T": T,
    "nb_simu": nb_simu,
    "random_state": random_state,
    "K": K,
    "baseline": pos,
    "conservative_levels": conservative_level,
}



algorithms = {
  'UCB': lambda T, MAB: UCB1(T, MAB, alpha=1),
 #   'UCBV': lambda T, MAB: UCBV(T, MAB),
#    'BootstrapUCB' : lambda T, MAB: BootstrapedUCB(T, MAB, delta = 0.1, b_rep = 200),
   'PHE' : lambda T, MAB : PHE(T, MAB, alpha =2),
#    'RE' : lambda T, MAB : Random_exploration(T, MAB, alpha = 3, verbose = False)
}

conservative_algorithms =  {
  'CUCB' : lambda T, MAB : CUCB(T, MAB, baseline, mean_baseline, conservative_level= conservative_level, oracle = False, version = 'old', batched = False, check_every = check_every, alpha = 1),
  'Oracle UCB' : lambda T, MAB : CUCB(T, MAB, baseline, mean_baseline, conservative_level= conservative_level, oracle = True, version = 'old', batched = False, check_every = check_every, alpha = 1),
#   'CUCB-new' : lambda T, MAB : CUCB(T, MAB, baseline, mean_baseline, conservative_level= conservative_level, oracle = False, version = 'new', batched = False, check_every = check_every, alpha = 1),
#  'CPHE-new' : lambda T, MAB : CPHE(T, MAB, baseline, mean_baseline, conservative_level = conservative_level, param_a1 = 2, version = 'new', batched = False, check_every = check_every),
  'CPHE' : lambda T, MAB : CPHE(T, MAB, baseline, mean_baseline, conservative_level = conservative_level, param_a1 = 2, version = 'old', batched = False, check_every = None),
#  'CPHE-oracle' : lambda T, MAB : CPHE(T, MAB, baseline, mean_baseline, conservative_level = conservative_level, param_a1 = 2, oracle = True),
#'SafetySetUCBV-old' : lambda T, MAB : SafetySetUCBV(T, MAB, baseline, mean_baseline, alpha=1., conservative_level= conservative_level, version ='old'),
#'SafetySetUCBV-new' : lambda T, MAB : SafetySetUCBV(T, MAB, baseline, mean_baseline, alpha=1., conservative_level= conservative_level, version = 'new')
}


results = []
full_algo = {**algorithms, **conservative_algorithms}


for alg_name in full_algo.keys():

    alg = full_algo[alg_name]

    regret = np.zeros((nb_simu, T))
    rwds = 0*regret 

    for k in tqdm(range(nb_simu), desc='Simulating {}'.format(alg_name)):
        if alg_name in ['SafetySetUCBV-old', 'SafetySetUCBV-new'] :
            rewards, draws, safe = alg(T, MAB)
        else :
            rewards, draws = alg(T, MAB)
        regret[k] = max(means) * np.arange(1, T + 1) - np.cumsum(rewards)
        rwds[k] = np.cumsum(means[draws.astype('int')])
    results += [(alg_name, MABResults(regret=regret, cum_rewards= rwds))]

id = '{:%Y%m%d_%H%M%S}_{}'.format(datetime.datetime.now(), 'GLM')
with open("{}_{}_MAB_illustration.pickle".format(id, "SEQ"), "wb") as f:
    pickle.dump(results, f)
with open("{}_{}_MAB_illustration_settings.json".format(id, "SEQ"), "w+") as f:
    json.dump(settings, f)
#%%
#plt.figure(1,figsize=(10, 10))
plt.figure(2,figsize=(10, 10))
t = np.arange(1, T+1)
for alg_name, val in results:
    mean_regret = np.mean(val.regret, axis=0)
    low_quantile = np.quantile(val.regret, 0.25, axis=0)
    high_quantile = np.quantile(val.regret, 0.75, axis=0)
    rwds = np.mean(val.cum_rewards, axis = 0)
    low_quantile_rwds = np.quantile(val.cum_rewards, 0.25, axis=0)
    high_quantile_rwds = np.quantile(val.cum_rewards, 0.75, axis=0)
#    
#    plt.figure(1)
#    plt.title('Margin')
#    temp = rwds - (1- conservative_level)*t*mean_baseline
#    plt.plot(temp[:200], label = alg_name)
#    plt.legend()
#    plt.fill_between(t, low_quantile_rwds - (1- conservative_level)*t*mean_baseline, high_quantile_rwds - (1- conservative_level)*t*mean_baseline, alpha = 0.15)
    print(alg_name, '=', min(rwds - (1- conservative_level)*t*mean_baseline))
    plt.figure(2)
    plt.title('Regret')
    plt.plot(mean_regret, label=alg_name)
    plt.legend()
    plt.fill_between(t, low_quantile, high_quantile, alpha = 0.15)

plt.show()
#import tikzplotlib
#tikzplotlib.save("lcb_worst.tex")

# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 15:37:36 2019

@author: evrard
"""


filename = '20190829_104727_linear_PAR_linear_results.pickle'
import pickle
import numpy as np
import pylab as plt

with open(filename, 'rb') as f:
        results = pickle.load(f)
        
n_models = len(results)
n_algos = len(results[0])
nb_simu,T = results[0][1][0][1]['regret'].shape
q = 0.25
t = np.linspace(1, T,T)
algorithms = {}
for alg_name, res in results[0][1] :
    algorithms[alg_name] = {'regret' : np.zeros((n_models,nb_simu, T)), 
              'cum_rewards' : np.zeros((n_models,nb_simu, T)),
              'norm_errors' : np.zeros((n_models,nb_simu, T))}
for m in range(n_models) :
    res = results[m][1]
    for i,val in enumerate(res) :
        alg_name = val[0]
        val = val[1]
        algorithms[alg_name]['regret'][m,:,:] = val['regret']
        algorithms[alg_name]['cum_rewards'][m,:,:] = val['cum_rewards']
        algorithms[alg_name]['norm_errors'][m,:,:] = val['norm_errors']

plt.figure(figsize = (10,10))
for alg_name, res in algorithms.items() :
    res['regret'] = res['regret'].cumsum(axis = 2)
    mean_regret = np.mean(res['regret'], axis = (0,1))
    low_quantile = np.quantile(res['regret'], q, axis = (0,1))
    high_quantile = np.quantile(res['regret'], 1-q, axis = (0,1))
    plt.plot(mean_regret, label = alg_name)
    plt.fill_between(t, low_quantile, high_quantile, alpha = 0.15)
    
plt.legend()




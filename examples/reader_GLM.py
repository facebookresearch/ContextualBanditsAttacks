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


filename = '/Users/evrardgarcelon/Desktop/monotone_mabs/20190825_113802_GLM_PAR_GLM_results.pickle'
import dill
import numpy as np
import pylab as plt

with open(filename, 'rb') as f:
        results = dill.load(f)
        
n_models = 1
n_algos = len(results[0])
nb_simu,T = results[0][1][0][1]['regret'].shape
clevels = results[-1]
baseline_means = np.zeros(n_models)
q = 0.25
t = np.linspace(0, T-1,T, dtype = 'int')
nb_first_iteration = 50
algorithms = {}
true_alg_name = {'GLM-UCB': 'UCB-GLM', 'GLM-CUCB-0.1': 'CUCB-GLM-0.1'}
for alg_name, res in results[0][1] :
    algorithms[true_alg_name[alg_name]] = {'regret' : np.zeros((n_models,nb_simu, T)), 
                  'cum_rewards' : np.zeros((n_models,nb_simu, T)),
                  'norm_errors' : np.zeros((n_models,nb_simu, T))}
for m in range(n_models) :
    res = results[m][1]
    baseline_means[m] = results[m][-1]
    for i,val in enumerate(res) :
        alg_name = val[0]
        val = val[1]
        algorithms[true_alg_name[alg_name]]['regret'][m,:,:] = val['regret']
        algorithms[true_alg_name[alg_name]]['cum_rewards'][m,:,:] = val['cum_rewards']
        algorithms[true_alg_name[alg_name]]['norm_errors'][m,:,:] = val['norm_errors']
        
plt.figure(1, figsize = (10,10))
plt.figure(2, figsize = (10,10))
regret = {}
margin = {}
for alg_name, res in algorithms.items() :
    temp = res['regret'].cumsum(axis = 2)
    mean_regret = np.mean(temp, axis = (0,1))
    low_quantile = np.quantile(temp,q, axis = (0,1))
    high_quantile = np.quantile(temp, 1-q, axis = (0,1))
    regret[alg_name] = (mean_regret, low_quantile, high_quantile)
    plt.figure(1)
    plt.plot(mean_regret, label = alg_name)
    plt.fill_between(t, low_quantile, high_quantile, alpha = 0.15)
    if alg_name != 'UCB-GLM' :
        res['cum_rewards'] = res['cum_rewards'].cumsum(axis = 2)    
        mean_margin = np.mean(res['cum_rewards'], axis = (0,1))
        low_quantile = np.quantile(res['cum_rewards'], q, axis = (0,1))
        high_quantile = np.quantile(res['cum_rewards'], 1-q, axis = (0,1))
        margin[alg_name] = (mean_margin, low_quantile, high_quantile)
    else :
        for alpha in clevels :
            a_name = alg_name + '-{}'.format(alpha)
            temp = 1*algorithms[alg_name]['cum_rewards']
            for m in range(n_models) :
                temp[m] = temp[m] - (1-alpha)*baseline_means[m]
            temp = temp.cumsum(axis = 2)    
            mean_margin = np.mean(temp, axis = (0,1))
            low_quantile = np.quantile(temp, q, axis = (0,1))
            high_quantile = np.quantile(temp, 1-q, axis = (0,1))
            margin[a_name] = (mean_margin[:nb_first_iteration], low_quantile[nb_first_iteration], high_quantile[nb_first_iteration])
    plt.figure(2)
    plt.plot(mean_margin[:nb_first_iteration], label = alg_name)
    plt.fill_between(t[:nb_first_iteration], low_quantile[:nb_first_iteration], high_quantile[:nb_first_iteration], alpha = 0.15)


plt.figure(2)
plt.plot(t[:nb_first_iteration], np.zeros(nb_first_iteration), color = 'red', linestyle = '--', label = '0')


plt.figure(1)
plt.legend()

plt.figure(2)
plt.legend()
plt.show()



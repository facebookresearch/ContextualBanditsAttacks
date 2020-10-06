# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import tikzplotlib
import os
import shutil
import sys
from cycler import cycler
import tarfile
import json
import re
sys.path.append('/private/home/broz/workspaces/bandits_attacks')


def tardir(path, tar_name):
    with tarfile.open(tar_name, "w:gz") as tar_handle:
        for root, dirs, files in os.walk(path):
            for file in files:
                tar_handle.add(os.path.join(root, file))


def get_eps(name):
    temp_eps = re.findall(r'[\d\.\d]+', name)
    temp_eps = np.array(list(map(float, temp_eps)))
    temp_eps = temp_eps[temp_eps <= 1]
    temp_eps = temp_eps[0]
    return temp_eps


def get_name(name):
    first, rest = name.split(' ', 1)
    return first


n = 9  # Number of colors
new_colors = [plt.get_cmap('Set1')(1. * i / n) for i in range(n)]
linestyle_cycler = cycler('linestyle', ['-', '--', ':', '-.', '-', '--', ':', '-.', '-'])
plt.rc('axes', prop_cycle=(cycler('color', new_colors) + linestyle_cycler))
plt.rc('lines', linewidth=2)

if len(sys.argv) == 2:
    filename = sys.argv[1]
else:
    filename = '20200121_153844_PAR_contextual_attacks_rewards.pickle'
with open(filename, 'rb') as f:
    results = pickle.load(f)

print("Opening file %s..." % filename)

setting_name = filename[:-7] + '_settings.json'
print('Opening settings %s...' % setting_name)

with open(setting_name, 'r') as f:
    settings = json.load(f)

folder = filename.split('.')[0]
if os.path.exists(folder):
    shutil.rmtree(folder)
os.makedirs(folder)
print("Done.\n")

with open(os.path.join(folder, setting_name), 'w') as f:
    json.dump(settings, f)

EVERY = 500
LW = 2
LATEX = True
nb_models = settings["nb_models"]
nb_simu = settings["nb_simu"]
real_T = settings["T"]
frequency = settings['frequency']
T = real_T // frequency
attack_parameter = settings["epsilon_tested"]
eps_plot_regret = attack_parameter[np.random.randint(low=0, high=len(attack_parameter))]

print("Generating regret and cost figures ...")
# select "bad" model
algorithms = {}
attacked_algorithms = {}
stationary_alg = {}
for alg_name, res in results[0][1]:
    if not 'attacked' in alg_name:
        algorithms[alg_name] = {'regret': np.zeros((nb_models, nb_simu, T)),
                                'cost': np.zeros((nb_models, nb_simu, T)),
                                'target_draws': np.zeros((nb_models, nb_simu, T)),
                                'rewards_range': np.zeros((nb_models, nb_simu, T))}
    elif 'gamma' in alg_name:
        eps = get_eps(alg_name)
        shortened_alg_name = get_name(alg_name)
        attacked_algorithms[(shortened_alg_name, eps)] = {'regret': np.zeros((nb_models, nb_simu, T)),
                                                          'cost': np.zeros((nb_models, nb_simu, T)),
                                                          'target_draws': np.zeros((nb_models, nb_simu, T)),
                                                          'rewards_range': np.zeros((nb_models, nb_simu, T))}
    else:
        pass
        shortened_alg_name = get_name(alg_name)
        stationary_alg[shortened_alg_name] = {'regret': np.zeros((nb_models, nb_simu, T)),
                                                          'cost': np.zeros((nb_models, nb_simu, T)),
                                                          'target_draws': np.zeros((nb_models, nb_simu, T)),
                                                          'rewards_range': np.zeros((nb_models, nb_simu, T))}
for m in range(nb_models):
    res = results[m][1]
    for i, val in enumerate(res):
        alg_name = val[0]
        val = val[1]
        print(val['regret'])
        if not 'attacked' in alg_name:
            algorithms[alg_name]['regret'][m, :, :] = val['regret']
            algorithms[alg_name]['cost'][m, :, :] = val['attack_cond']
            algorithms[alg_name]['target_draws'][m, :, :] = val['target_draws']
            algorithms[alg_name]['rewards_range'][m, :, :] = val['range_rewards']
        elif 'gamma' in alg_name:
            eps = get_eps(alg_name)
            shortened_alg_name = get_name(alg_name)
            attacked_algorithms[(shortened_alg_name, eps)]['regret'][m, :, :] = val['regret']
            attacked_algorithms[(shortened_alg_name, eps)]['cost'][m, :, :] = val['attack_cond']
            attacked_algorithms[(shortened_alg_name, eps)]['target_draws'][m, :, :] = val['target_draws']
            attacked_algorithms[(shortened_alg_name, eps)]['rewards_range'][m, :, :] = val['range_rewards']
        else:
            shortened_alg_name = get_name(alg_name)
            stationary_alg[shortened_alg_name]['regret'][m, :, :] = val['regret']
            stationary_alg[shortened_alg_name]['cost'][m, :, :] = val['attack_cond']
            stationary_alg[shortened_alg_name]['target_draws'][m, :, :] = val['target_draws']
            stationary_alg[shortened_alg_name]['rewards_range'][m, :, :] = val['range_rewards']


plt.figure(1)
t = np.linspace(0, T - 1, T, dtype='int') * frequency
rep = nb_models * nb_simu
for alg_name, res in algorithms.items():

    #Plot the regret ofr the normal alg
    res['regret'] = res['regret'].cumsum(axis=2)
    mean_regret = np.mean(res['regret'], axis=(0, 1))
    std = np.std(res['regret'], axis=(0, 1))/np.sqrt(rep)
    plt.plot(t[::EVERY], mean_regret[::EVERY], linewidth=LW, label=alg_name)
<<<<<<< HEAD
=======
    plt.fill_between(t[::EVERY], mean_regret[::EVERY] - 2 * std[::EVERY], mean_regret[::EVERY] + 2 * std[::EVERY],
                     alpha=0.15)
    #Plot the regret for the attacked algorithms
    regret = attacked_algorithms[(alg_name, eps_plot_regret)]['regret'].cumsum(axis=2)
    mean_regret = np.mean(regret, axis=(0, 1))
    std = np.std(regret, axis=(0, 1))/np.sqrt(rep)
    plt.plot(t[::EVERY], mean_regret[::EVERY], linewidth=LW, label=alg_name + ' attacked eps {:.2f}'.format(eps_plot_regret))
    plt.fill_between(t[::EVERY], mean_regret[::EVERY] - 2 * std[::EVERY], mean_regret[::EVERY] + 2 * std[::EVERY],
                     alpha=0.15)

    regret = stationary_alg[alg_name]['regret'].cumsum(axis=2)
    mean_regret = np.mean(regret, axis=(0, 1))
    std = np.std(regret, axis=(0, 1))/np.sqrt(rep)
    plt.plot(t[::EVERY], mean_regret[::EVERY], linewidth=LW, label=alg_name + ' attacked statinary')
>>>>>>> 443bf801ec40dea0146947420af027f021b809a6
    plt.fill_between(t[::EVERY], mean_regret[::EVERY] - 2 * std[::EVERY], mean_regret[::EVERY] + 2 * std[::EVERY],
                     alpha=0.15)
    #Plot the regret for the attacked algorithms
    # regret = attacked_algorithms[(alg_name, eps_plot_regret)]['regret'].cumsum(axis=2)
    # mean_regret = np.mean(regret, axis=(0, 1))
    # std = np.std(regret, axis=(0, 1))/np.sqrt(rep)
    # plt.plot(t[::EVERY], mean_regret[::EVERY], linewidth=LW, label=alg_name + ' attacked eps {:.2f}'.format(eps_plot_regret))
    # plt.fill_between(t[::EVERY], mean_regret[::EVERY] - 2 * std[::EVERY], mean_regret[::EVERY] + 2 * std[::EVERY],
    #                  alpha=0.15)
    #
    # regret = stationary_alg[alg_name]['regret'].cumsum(axis=2)
    # mean_regret = np.mean(regret, axis=(0, 1))
    # std = np.std(regret, axis=(0, 1))/np.sqrt(rep)
    # plt.plot(t[::EVERY], mean_regret[::EVERY], linewidth=LW, label=alg_name + ' attacked statinary')
    # plt.fill_between(t[::EVERY], mean_regret[::EVERY] - 2 * std[::EVERY], mean_regret[::EVERY] + 2 * std[::EVERY],
    #                  alpha=0.15)

plt.title('Cumulative regret')
plt.legend()
plt.savefig(os.path.join(folder, "avg_regret.png"))
if LATEX:
    tikzplotlib.save(os.path.join(folder, "avg_regret.tex"))

plt.figure(2)
for alg_name, res in algorithms.items():

    # #Plot the regret ofr the normal alg
    # res['cost'] = res['cost'].cumsum(axis=2)
    # mean_cost = np.mean(res['cost'], axis=(0, 1))
    # std = np.std(res['cost'], axis=(0, 1))/np.sqrt(rep)
    # plt.plot(t[::EVERY], mean_cost[::EVERY], linewidth=LW, label=alg_name)
    # plt.fill_between(t[::EVERY], mean_cost[::EVERY] - 2 * std[::EVERY], mean_cost[::EVERY] + 2 * std[::EVERY],
    #                  alpha=0.15)
    #Plot the regret for the attacked algorithms
    cost = attacked_algorithms[(alg_name, eps_plot_regret)]['cost'].cumsum(axis=2)
    mean_cost = np.mean(cost, axis=(0, 1))
    std = np.std(cost, axis=(0, 1))/np.sqrt(rep)
    plt.plot(t[::EVERY], mean_cost[::EVERY], linewidth=LW, label=alg_name + ' attacked eps {:.2f}'.format(eps_plot_regret))
    plt.fill_between(t[::EVERY], mean_cost[::EVERY] - 2 * std[::EVERY], mean_cost[::EVERY] + 2 * std[::EVERY],
                     alpha=0.15)

    cost = stationary_alg[alg_name]['cost'].cumsum(axis=2)
    mean_cost = np.mean(cost, axis=(0, 1))
    std = np.std(cost, axis=(0, 1))/np.sqrt(rep)
    plt.plot(t[::EVERY], mean_cost[::EVERY], linewidth=LW, label=alg_name + ' attacked stationary')
    plt.fill_between(t[::EVERY], mean_cost[::EVERY] - 2 * std[::EVERY], mean_cost[::EVERY] + 2 * std[::EVERY],
                     alpha=0.15)

plt.title('Total attack cost')
plt.legend()
plt.savefig(os.path.join(folder, "avg_cost.png"))
if LATEX:
    tikzplotlib.save(os.path.join(folder, "avg_cost.tex"))

plt.figure(3)
for alg_name, res in algorithms.items():

    # #Plot the regret ofr the normal alg
    # res['target_draws'] = res['target_draws'].cumsum(axis=2)
    # mean_draws = np.mean(res['target_draws'], axis=(0, 1))
    # std = np.std(res['target_draws'], axis=(0, 1))/np.sqrt(rep)
    # plt.plot(t[::EVERY], mean_draws[::EVERY], linewidth=LW, label=alg_name)
    # plt.fill_between(t[::EVERY], mean_draws[::EVERY] - 2 * std[::EVERY], mean_draws[::EVERY] + 2 * std[::EVERY],
    #                  alpha=0.15)

    draws = attacked_algorithms[(alg_name, eps_plot_regret)]['target_draws'].cumsum(axis=2)
    mean_draws = np.mean(draws, axis=(0, 1))
    std = np.std(draws, axis=(0, 1))/np.sqrt(rep)
    plt.plot(t[::EVERY], mean_draws[::EVERY], linewidth=LW, label=alg_name + ' attacked eps {:.2f}'.format(eps_plot_regret))
    plt.fill_between(t[::EVERY], mean_draws[::EVERY] - 2 * std[::EVERY], mean_draws[::EVERY] + 2 * std[::EVERY],
                     alpha=0.15)

    draws = stationary_alg[alg_name]['target_draws'].cumsum(axis=2)
    mean_draws = np.mean(draws, axis=(0, 1))
    std = np.std(draws, axis=(0, 1))/np.sqrt(rep)
    plt.plot(t[::EVERY], mean_draws[::EVERY], linewidth=LW, label=alg_name + ' attacked stationary'.format(eps_plot_regret))
    plt.fill_between(t[::EVERY], mean_draws[::EVERY] - 2 * std[::EVERY], mean_draws[::EVERY] + 2 * std[::EVERY],
                     alpha=0.15)

plt.title('Total target arm draws')
plt.legend()
plt.savefig(os.path.join(folder, "avg_draws.png"))
if LATEX:
    tikzplotlib.save(os.path.join(folder, "avg_draws.tex"))

print("Generating impact of epsilon figure")

regrets_mean = {}
costs_mean = {}
draws_mean = {}
regrets_std = {}
costs_std = {}
draws_std = {}
for alg_name in algorithms.keys():
    list_r_mean = []
    list_c_mean = []
    list_d_mean = []
    list_r_std = []
    list_c_std = []
    list_d_std = []
    for eps in attack_parameter:
        r = attacked_algorithms[(alg_name, eps)]['regret'].cumsum(axis=2)[:, :, -1]
        std_r = np.std(r)/np.sqrt(rep)
        mean_r = np.mean(r)
        c = attacked_algorithms[(alg_name, eps)]['cost'].cumsum(axis=2)[:, :, -1]
        std_c = np.std(c)/np.sqrt(rep)
        mean_c = np.mean(c)
        d = attacked_algorithms[(alg_name, eps)]['target_draws'].cumsum(axis=2)[:, :, -1]
        std_d = np.std(d)/np.sqrt(rep)
        mean_d = np.mean(d)
        list_r_mean.append(mean_r)
        list_c_mean.append(mean_c)
        list_d_mean.append(mean_d)
        list_r_std.append(std_r)
        list_c_std.append(std_c)
        list_d_std.append(std_d)
    regrets_mean[alg_name] = np.array(list_r_mean)
    costs_mean[alg_name] = np.array(list_c_mean)
    draws_mean[alg_name] = np.array(list_d_mean)
    regrets_std[alg_name] = np.array(list_r_std)
    costs_std[alg_name] = np.array(list_c_std)
    draws_std[alg_name] = np.array(list_d_std)

plt.figure(4)
plt.title('Cost as a function of attack parameter at T={}'.format(T))
for alg_name in algorithms.keys():
    c = costs_mean[alg_name]
    std = costs_std[alg_name]
    plt.plot(attack_parameter, c, linewidth=LW, label=alg_name)
    plt.fill_between(attack_parameter, c - 2 * std, c + 2 * std, alpha=0.15)
plt.legend()
plt.savefig(os.path.join(folder, "cost_epsilon.png"))

plt.figure(5)
plt.title('Regret as a function of attack parameter at T={}'.format(T))
for alg_name in algorithms.keys():
    r = regrets_mean[alg_name]
    std = regrets_std[alg_name]
    plt.plot(attack_parameter, r, linewidth=LW, label=alg_name)
    plt.fill_between(attack_parameter, r - 2 * std, r + 2 * std, alpha=0.15)
plt.legend()

plt.savefig(os.path.join(folder, "regret_epsilon.png"))

plt.figure(6)
plt.title('Target draws as a function of attack parameter at T={}'.format(T))
for alg_name in algorithms.keys():
    d = draws_mean[alg_name]
    std = draws_std[alg_name]
    plt.plot(attack_parameter, d, linewidth=LW, label=alg_name)
    plt.fill_between(attack_parameter, d - 2 * std, d + 2 * std, alpha=0.15)
plt.legend()

plt.savefig(os.path.join(folder, "draws_epsilon.png"))

for eps in attack_parameter:
    rewards = np.array([])
    for alg_name in algorithms.keys():
        rewards = np.concatenate((rewards, attacked_algorithms[(alg_name, eps)]['rewards_range']), axis=None)
    print('-'*100)
    print('The maximum reward for epsilon = {:.2f} is:'.format(eps), np.max(rewards))
    print('The minimum reward for epsilon = {:.2f} is:'.format(eps), np.min(rewards))
    print('The mean reward for epsilon = {:.2f} is:'.format(eps), np.mean(rewards))
    print('The median reward for epsilon = {:.2f} is:'.format(eps), np.median(rewards))
    print('The 25% quantile reward for epsilon = {:.2f} is:'.format(eps), np.quantile(rewards, 0.25))
    print('The 75% quantile reward for epsilon = {:.2f} is:'.format(eps), np.quantile(rewards, 0.75))
    print('The perctange reward over 1 for epsilon = {:.2f} is:'.format(eps), np.sum(rewards > 1)/len(rewards))
    print('The perctange reward below 0 for epsilon = {:.2f} is:'.format(eps), np.sum(rewards < 0) / len(rewards))


# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 23:30:58 2019

@author: evrardgarcelon
"""

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


def tardir(path, tar_name):
    with tarfile.open(tar_name, "w:gz") as tar_handle:
        for root, dirs, files in os.walk(path):
            for file in files:
                tar_handle.add(os.path.join(root, file))


n = 9  # Number of colors
new_colors = [plt.get_cmap('Set1')(1. * i / n) for i in range(n)]
linestyle_cycler = cycler('linestyle', ['-', '--', ':', '-.', '-', '--', ':', '-.', '-'])
plt.rc('axes', prop_cycle=(cycler('color', new_colors) + linestyle_cycler))
plt.rc('lines', linewidth=2)

if len(sys.argv) == 2:
    filename = sys.argv[1]
else:
    filename = '20190822_065452_Bernoulli_PAR_martingale_results.pickle'
    filename = '20190901_124136_linear_PAR_linear_results.pickle'
    filename = '20190902_135139_linear_PAR_linear_results.pickle'
    filename = '20190903_233609_linear_PAR_jester_res§ults.pickle'
    filename = '20190903_235606_linear_PAR_jester_results.pickle'
    filename = '20190904_010618_linear_PAR_jester_results.pickle'

with open(filename, 'rb') as f:
    results = pickle.load(f)

print("Opening file %s..." % filename)
folder = filename.split('.')[0]
if os.path.exists(folder):
    shutil.rmtree(folder)
os.makedirs(folder)
print("Done.\n")

EVERY = 20
LW = 2

print("Generating all figures ...")
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


def tardir(path, tar_name):
    with tarfile.open(tar_name, "w:gz") as tar_handle:
        for root, dirs, files in os.walk(path):
            for file in files:
                tar_handle.add(os.path.join(root, file))



<<<<<<< HEAD
=======
with open(filename, 'rb') as f:
    results = pickle.load(f)

print("Opening file %s..." % filename)
folder = filename.split('.')[0]
if os.path.exists(folder):
    shutil.rmtree(folder)
os.makedirs(folder)
print("Done.\n")

EVERY = 200
LW = 2
>>>>>>> e314f8f8accdff7717898e2745f92c6c0e230275

print("Generating all figures ...")
bad_model = None
min_val = np.inf
total_experiments = {}
avg_area = {}
avg_margin = {}
for m, model in enumerate(results):
    cucb_M, cucb_H = 0, 0
    plt.figure()
    ymax = -np.inf
    T = None
    for alg_name, val in model[1]:

        if alg_name not in total_experiments.keys():
            total_experiments[alg_name] = []
            avg_area[alg_name] = []
            avg_margin[alg_name] = []

        rep, T = val['cum_rewards'].shape

        t = np.arange(1, T + 1)
        regret = np.cumsum(val['regret'], axis=1)
        total_experiments[alg_name] += regret.tolist()

        margin = val['cum_rewards'].cumsum(axis=1)
        area = np.sum(margin * (margin < 0), axis=1).mean()
        print('min_margin(', alg_name, ')=', margin.min())
        print('area(', alg_name, ')=', area)
        print()
        avg_area[alg_name] += [area]
        avg_margin[alg_name] += margin.tolist()

        mean_regret = np.mean(regret, axis=0)
        std = np.std(regret, axis=0) / np.sqrt(rep)
        plt.plot(t[::EVERY], mean_regret[::EVERY], linewidth=LW, label=alg_name)
        plt.fill_between(t[::EVERY], mean_regret[::EVERY] - 2 * std[::EVERY], mean_regret[::EVERY] + 2 * std[::EVERY],
                         alpha=0.15)
        ymax = max(ymax, mean_regret[-1] + 2 * std[-1])

    plt.xlim([0, T])
    plt.ylim([0, ymax])
    plt.legend()
    plt.title("model: {}".format(m))
    plt.savefig(os.path.join(folder, "model{}.png".format(m)))
    tikzplotlib.save(os.path.join(folder, "model{}.tex".format(m)))
    plt.close()

print("Done.\n")

avg_regret_name = os.path.join(folder, "avg_regret.png")
print("Saving average regret to %s..." % avg_regret_name)
ymax = -np.inf
TOSAVE = {}
for alg_name in total_experiments.keys():
    regret = np.array(total_experiments[alg_name])
    rep, T = regret.shape
    t = np.arange(1, T + 1)
    mean_regret = np.mean(regret, axis=0)
    std = np.std(regret, axis=0) / np.sqrt(rep)
    plt.plot(t[::EVERY], mean_regret[::EVERY], linewidth=LW, label=alg_name)
    plt.fill_between(t[::EVERY],
                     mean_regret[::EVERY] - 2 * std[::EVERY], mean_regret[::EVERY] + 2 * std[::EVERY],
                     alpha=0.15)
    ymax = max(ymax, mean_regret[-1] + 2 * std[-1])

    M = np.concatenate((t.reshape(-1, 1), mean_regret.reshape(-1, 1), std.reshape(-1, 1)), axis=1)
    TOSAVE[alg_name] = M

np.savez_compressed(os.path.join(folder, "avg_regret"), **TOSAVE)

plt.xlim([0, T])
plt.ylim([0, ymax])
plt.xlabel("Time")
plt.ylabel("Cumulative Regret")
plt.legend()
plt.savefig(avg_regret_name)
plt.close()
print("Done.\n")

for alg_name in avg_area.keys():
    print("AverageAREA({}): {}".format(alg_name, np.mean(avg_area[alg_name])))


avg_margin_name = os.path.join(folder, "avg_margin.png")
print("Saving average margin to %s..." % avg_margin_name)
plt.figure(figsize=(10,10))
ymax = -np.inf
maxt = -np.inf
TOSAVE = {}
for alg_name in avg_margin.keys():
    margin = np.array(avg_margin[alg_name])
    rep, T = margin.shape
    t = np.arange(1, T + 1)
    mean_margin = np.mean(margin, axis=0)
    std = np.std(margin, axis=0) / np.sqrt(rep)
    idxs = mean_margin < 10
    if np.sum(idxs) > 0:
        plt.plot(t[idxs], mean_margin[idxs], linewidth=LW, label=alg_name)
        plt.fill_between(t[idxs],
                         mean_margin[idxs] - 2 * std[idxs], mean_margin[idxs] + 2 * std[idxs],
                         alpha=0.15)
        ymax = max(ymax, mean_margin[-1] + 2 * std[-1])
        maxt = max(maxt, np.max(t[idxs]))

    M = np.concatenate((t.reshape(-1,1), mean_margin.reshape(-1,1), std.reshape(-1,1)), axis=1)
    TOSAVE[alg_name] = M

plt.xlim([0, maxt])
# plt.ylim([0, ymax])
plt.xlabel("Time")
plt.ylabel("Average Margin")
plt.legend()
plt.savefig(avg_margin_name)
plt.close()

np.savez_compressed(os.path.join(folder, "avg_margin"), **TOSAVE)
print("Done.\n")

# worst_name = os.path.join(folder, "worst_linear_exp.tex")
# print("Saving worst model to %s..." % worst_name)
# tikzplotlib.save(worst_name)
# print("Done.\n")

archive_name = "{}.tar.gz".format(folder)
print("Compressing files to %s..." % archive_name)
tardir(folder, archive_name)
print("Done.\n")

plt.show()

# # select "bad" model
# plt.figure(1,figsize=(10, 10))
# plt.figure(1)
# plt.clf()
# plt.title('Regret')
# ymax = -np.inf
# T = None
# for alg_name, res in results:
#     val = res[0][1]
#     print(alg_name)
#     rep, T = val['cum_rewards'].shape
#     t = np.arange(1, T + 1)
#     regret = np.cumsum(val['regret'], axis=1)
#     mean_regret = np.mean(regret, axis=0)
#     std = np.std(regret, axis=0) / np.sqrt(rep)
#     plt.plot(t[::EVERY], mean_regret[::EVERY], linewidth=LW, label=alg_name)
#     plt.fill_between(t[::EVERY], mean_regret[::EVERY] - 2 * std[::EVERY], mean_regret[::EVERY] + 2 * std[::EVERY],
#                      alpha=0.15)
#     ymax = max(ymax, mean_regret[-1] + 2 * std[-1])
#
# plt.xlim([0, T])
# plt.ylim([0, ymax])
# plt.legend()
# plt.savefig(os.path.join(folder, "real_data.png"))
# print("Done.\n")
#
# worst_name = os.path.join(folder, "real_data.tex")
# print("Saving worst model to %s..." % worst_name)
# tikzplotlib.save(worst_name)
#
# plt.figure(2,figsize=(10, 10))
# plt.figure(2)
# plt.clf()
# plt.title('Margin')
# ymax = -np.inf
# max_time = 1000
# T = None
# for alg_name, res in results:
#     val = res[0][1]
#     rep, T = val['cum_rewards'].shape
#     t = np.arange(1, T + 1)
#     margin = val['cum_rewards'].cumsum(axis = 1)
#     print(alg_name, '=', margin.min())
#     mean_margin = np.mean(margin, axis=0)
#     std = np.std(margin, axis=0) / np.sqrt(rep)
#     plt.plot(t[:max_time:EVERY], mean_margin[:max_time:EVERY], linewidth=LW, label=alg_name)
#     plt.fill_between(t[:max_time:EVERY], mean_margin[:max_time:EVERY] - 2 * std[:max_time:EVERY], mean_margin[:max_time:EVERY] + 2 * std[:max_time:EVERY],
#                      alpha=0.15)
#     #ymax = max(ymax, mean_regret[-1] + 2 * std[-1])
#
# plt.xlim([0, max_time])
# #plt.ylim([0, ymax])
# plt.legend()
# plt.savefig(os.path.join(folder, "real_data_margin.png"))
# print("Done.\n")
#
# worst_name = os.path.join(folder, "real_data.tex")
# print("Saving worst model to %s..." % worst_name)
# tikzplotlib.save(worst_name)
# print("Done.\n")
#
# archive_name = "{}.tar.gz".format(folder)
# print("Compressing files to %s..." % archive_name)
# tardir(folder, archive_name)
# print("Done.\n")
#
# plt.show()

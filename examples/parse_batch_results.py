#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Created on Thu Aug 22 15:37:36 2019

@author: evrard
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib
import os
import sys
import shutil
from cycler import cycler
import tarfile


def tardir(path, tar_name):
    with tarfile.open(tar_name, "w:gz") as tar_handle:
        for root, dirs, files in os.walk(path):
            for file in files:
                tar_handle.add(os.path.join(root, file))


def plot_model(model, name):
    ymax = -np.inf
    batches = []

    plt.figure()
    plt.title('model: {}'.format(name))
    area = 0.
    for p, AAA in model[1]:
        batches = []
        UCB_vals = None
        UCB_stds = None
        CUCB1_vals = None
        CUCB1_stds = None
        CUCBB_vals = []
        CUCBB_stds = []
        for alg_name, mean_regret, std in AAA:
            print(alg_name)
            if alg_name == "UCB":
                UCB_vals = mean_regret[-1]
                UCB_stds = std[-1]
            elif alg_name == "CUCB-new-0.1-1":
                CUCB1_vals = mean_regret[-1]
                CUCB1_stds = std[-1]
                CUCBB_vals.append(mean_regret[-1])
                CUCBB_stds.append(std[-1])
                batches.append(int(alg_name.split('-')[-1]))
            else:
                CUCBB_vals.append(mean_regret[-1])
                CUCBB_stds.append(std[-1])
                batches.append(int(alg_name.split('-')[-1]))

        # area += CUCBB_vals - UCB_vals
        CUCBB_vals = np.array(CUCBB_vals)
        CUCBB_stds = np.array(CUCBB_stds)

        if CUCB1_vals is not None:
            ax1 = plt.plot([batches[0], batches[-1]], [CUCB1_vals, CUCB1_vals], label='CUCB_p{}'.format(p),
                           marker='o')
            ax1_col = ax1[0].get_color()
            plt.fill_between([batches[0], batches[-1]], CUCB1_vals - 2 * CUCB1_stds, CUCB1_vals + 2 * CUCB1_stds,
                             alpha=0.15, color=ax1_col)
            ymax = max(ymax, CUCB1_vals + 2 * CUCB1_stds)
        if UCB_vals is not None:
            ax1 = plt.plot([batches[0], batches[len(batches) - 1]], [UCB_vals, UCB_vals],
                           label='UCB_p{}'.format(p), marker='+')
            ax1_col = ax1[0].get_color()
            plt.fill_between(batches, UCB_vals - 2 * UCB_stds, UCB_vals + 2 * UCB_stds, alpha=0.15, color=ax1_col)
            ymax = max(ymax, UCB_vals + 2 * UCB_stds)
        if len(CUCBB_vals) > 0:
            ax1 = plt.plot(batches, CUCBB_vals, label='CUCB_BATCH_p{}'.format(p), marker='d')
            ax1_col = ax1[0].get_color()
            plt.fill_between(batches, CUCBB_vals - 2 * CUCBB_stds, CUCBB_vals + 2 * CUCBB_stds,
                             alpha=0.15, color=ax1_col)
            ymax = max(ymax, np.max(CUCBB_vals + 2 * CUCBB_stds))

    N = np.asscalar(np.max(batches))
    plt.xlim([0, N])
    plt.ylim([0, ymax])
    plt.xlabel("Checkpoint")
    plt.ylabel("Regret")
    plt.legend()

    return area


def plot_all2(results):

    processed_results = {}

    T = None
    for m, model in enumerate(results):
        batches = []

        for p, AAA in model[1]:

            if p not in processed_results.keys():
                processed_results[p] = []

            batches = []
            UCB_vals = None
            UCB_stds = None
            CUCB1_vals = None
            CUCB1_stds = None
            CUCBB_vals = []
            CUCBB_stds = []
            for alg_name, mean_regret, std in AAA:
                print(alg_name)
                if alg_name == "UCB":
                    T = len(mean_regret)
                    UCB_vals = mean_regret[-1]
                    UCB_stds = std[-1]
                elif alg_name == "CUCB-new-0.1-1":
                    CUCB1_vals = mean_regret[-1]
                    CUCB1_stds = std[-1]
                    CUCBB_vals.append(mean_regret[-1])
                    CUCBB_stds.append(std[-1])
                    batches.append(int(alg_name.split('-')[-1]))
                else:
                    CUCBB_vals.append(mean_regret[-1])
                    CUCBB_stds.append(std[-1])
                    batches.append(int(alg_name.split('-')[-1]))

            # area += CUCBB_vals - UCB_vals
            CUCBB_vals = np.array(CUCBB_vals)
            CUCBB_stds = np.array(CUCBB_stds)

            processed_results[p].append((CUCBB_vals - UCB_vals).tolist())

    for p in processed_results.keys():
        vals = np.array(processed_results[p])
        mean = np.mean(vals, axis=0)
        std = np.std(vals, axis=0) / np.sqrt(vals.shape[0])
        ax1 = plt.plot(batches, mean, label="p={}".format(p))
        ax1_col = ax1[0].get_color()
        plt.fill_between(batches, mean - 2 * std, mean + 2 * std, alpha=0.15, color=ax1_col)
    plt.legend()
    plt.xlabel("Checkpoint ($T$)")
    plt.ylabel("R_(CUCB2)({}) - R_(UCB)({})".format(T, T))


n = 9  # Number of colors
new_colors = [plt.get_cmap('Set1')(1. * i / n) for i in range(n)]
linestyle_cycler = cycler('linestyle', ['-', '--', ':', '-.', '-', '--', ':', '-.', '-'])
plt.rc('axes', prop_cycle=(cycler('color', new_colors) + linestyle_cycler))
plt.rc('lines', linewidth=2)

if len(sys.argv) == 1:
    filename = 'COMP_20190825_033627_batch_results.pickle'
else:
    filename = sys.argv[1]

SUMMARIZE = False

print("Opening file %s..." % filename)
with open(filename, 'rb') as f:
    results = pickle.load(f)
print("Done.\n")

folder = filename.split('.')[0]
if os.path.exists(folder):
    shutil.rmtree(folder)
os.makedirs(folder)

EVERY = 10
LW = 2

if SUMMARIZE:

    new_results = []
    for m, model in enumerate(results):
        BBB = []
        for pos, algorithms in model[1]:
            AAA = []
            for alg_name, val in algorithms:
                # print(m, pos, alg_name)
                rep, T = val['cum_rewards'].shape
                mean_regret = np.mean(val['regret'], axis=0)
                std = np.std(val['regret'], axis=0) / np.sqrt(rep)
                AAA.append((alg_name, mean_regret, std))
            BBB.append((pos, AAA))
        new_results.append((m, BBB, model[2]))

    with open("COMP_{}".format(filename), "wb") as f:
        pickle.dump(new_results, f)


else:
    bad_model = None
    max_area = -np.inf

    print("Generating all figures ...")

    plot_all2(results)
    # for m, model in enumerate(results):
    #
    #     area = plot_model(model, name=m)
    #     plt.savefig(os.path.join(folder, "model{}.png".format(m)))
    #     tikzplotlib.save(os.path.join(folder, "model{}.tex".format(m)))
    #     plt.close()
    #
    #     if area > max_area:
    #         bad_model = m
    #         max_area = area
    #         print(max_area)
    #
    # plot_model(results[bad_model], name=bad_model)
    # plt.savefig(os.path.join(folder, "worst_mab_exp.png"))
    # plt.close()
    # print("Done.\n")

    worst_name = os.path.join(folder, "mab_batch.tex")
    print("Saving worst model to %s ..." % worst_name)
    tikzplotlib.save(worst_name)
    print("Done.\n")

    archive_name = "{}.tar.gz".format(folder)
    print("Compressing files to %s ..." % archive_name)
    tardir(folder, archive_name)
    print("Done.\n")

    plt.show()

# n_models = len(results)
# n_batch = len(results[0][1]) - 1
# nb_simu, T = results[0][1][0][1]['regret'].shape
# batches = []
# q = 0.25
# regret_batch_at_T = np.zeros((n_models, n_batch, nb_simu))
# regret_UCB_T = np.zeros((n_models, 1, nb_simu))
# for m in range(n_models):
#     res = results[m][1]
#     for i, val in enumerate(res):
#         alg_name = val[0]
#         val = val[1]
#         if alg_name == 'UCB':
#             regret_UCB_T[m] = val['regret'][:, -1]
#         else:
#             alg_name[13::]
#             batches.append(int(alg_name[13::]))
#             regret_batch_at_T[m, i - 1, :] = val['regret'][:, -1]
#
# batches = np.array(batches)
# batches = batches / T
# regret_diff = regret_batch_at_T - regret_UCB_T
# mean_regret_diff = np.mean(regret_diff, axis=(0, 2))
# high_quantile = np.quantile(regret_diff, 1 - q, axis=(0, 2))
# low_quantile = np.quantile(regret_diff, q, axis=(0, 2))
# plt.plot(batches, mean_regret_diff, color='green')
# plt.fill_between(batches, low_quantile, high_quantile, alpha=0.15, color='green')
# plt.show()

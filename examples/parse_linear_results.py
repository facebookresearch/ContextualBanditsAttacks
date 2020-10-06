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

ALPHA = 0.05


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

with open(filename, 'rb') as f:
    results = pickle.load(f)

print("Opening file %s..." % filename)

setting_name = filename[:-14] + 'settings.json'
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

EVERY = 200
LW = 2
LATEX = True
SAVE_MARGIN_FOR_EACH_MODEL=True

print("Generating all figures ...")
# select "bad" model
fpoint = open(os.path.join(folder, "scores.txt"), "w")
bad_model = None
min_val = np.inf
total_experiments = {}
for m, model in enumerate(results):
    cucb_M, cucb_H = 0, 0
    plt.figure()
    ymax = -np.inf
    T = None
    for alg_name, val in model[1]:
        rep, T = val['cum_rewards'].shape

        if alg_name not in total_experiments.keys():
            total_experiments[alg_name] = []

        t = np.arange(1, T + 1)
        regret = np.cumsum(val['regret'], axis=1)
        mean_regret = np.mean(regret, axis=0)
        std = np.std(regret, axis=0) / np.sqrt(rep)
        plt.plot(t[::EVERY], mean_regret[::EVERY], linewidth=LW, label=alg_name)
        plt.fill_between(t[::EVERY], mean_regret[::EVERY] - 2 * std[::EVERY], mean_regret[::EVERY] + 2 * std[::EVERY],
                         alpha=0.15)
        ymax = max(ymax, mean_regret[-1] + 2 * std[-1])

        total_experiments[alg_name] += regret.tolist()

        if alg_name in ["CLUCB-new-{}-1".format(ALPHA)]:
            rep, T = val['cum_rewards'].shape
            regret = np.cumsum(val['regret'], axis=1)
            mean_regret = np.mean(regret, axis=0)
            std = np.std(regret, axis=0) / np.sqrt(rep)
            cucb_M = mean_regret[-1] + 2 * std[-1]
        if alg_name in ["CLUCB-old-{}-1".format(ALPHA)]:
            rep, T = val['cum_rewards'].shape
            regret = np.cumsum(val['regret'], axis=1)
            mean_regret = np.mean(regret, axis=0)
            std = np.std(regret, axis=0) / np.sqrt(rep)
            cucb_H = mean_regret[-1] - 2 * std[-1]
    val = abs(cucb_M - cucb_H) / cucb_H
    print(m, val)
    fpoint.write("{} {}\n".format(m, val))
    if val < min_val:
        bad_model = m
        min_val = val

    plt.xlim([0, T])
    plt.ylim([0, ymax])
    plt.legend()
    plt.title("model: {}".format(m))
    plt.savefig(os.path.join(folder, "model{}.png".format(m)))
    if LATEX:
        tikzplotlib.save(os.path.join(folder, "model{}.tex".format(m)))
    plt.close()
fpoint.close()

print("Generating all figures for margin ...")
avg_area = {}
avg_margin = {}
for m, model in enumerate(results):
    plt.figure()
    ymax = -np.inf
    ymin = np.inf
    maxt = 0
    T = None
    print()

    TOSAVE = {}

    for alg_name, val in model[1]:

        if alg_name not in avg_area.keys():
            avg_area[alg_name] = []
            avg_margin[alg_name] = []

        rep, T = val['cum_rewards'].shape
        margin = val['cum_rewards'].cumsum(axis=1)
        t = np.arange(1, T + 1)

        area = np.sum(margin * (margin < 0), axis=1).mean()
        print('min_margin(', alg_name, ')=', margin.min())
        print('area(', alg_name, ')=', area)

        mean_margin = np.mean(margin, axis=0)
        std = np.std(margin, axis=0) / np.sqrt(rep)

        if SAVE_MARGIN_FOR_EACH_MODEL:
            M = np.concatenate((t.reshape(-1, 1), mean_margin.reshape(-1, 1), std.reshape(-1, 1)), axis=1)
            TOSAVE[alg_name] = M

        avg_area[alg_name] += [area]
        avg_margin[alg_name] += margin.tolist()

        idxs = mean_margin < 10
        if np.sum(idxs) > 0:
            plt.plot(t[idxs], mean_margin[idxs], linewidth=LW, label=alg_name)
            plt.fill_between(t[idxs],
                             mean_margin[idxs] - 2 * std[idxs], mean_margin[idxs] + 2 * std[idxs],
                             alpha=0.15)
            ymin = min(ymin, np.min(mean_margin[idxs] - 2 * std[idxs]))
            ymax = max(ymax, np.max(mean_margin[idxs] + 2 * std[idxs]))
            maxt = max(maxt, np.max(t[idxs]))

    if SAVE_MARGIN_FOR_EACH_MODEL:
        np.savez_compressed(os.path.join(folder, "model{}_margin".format(m)), **TOSAVE)

    plt.xlim([1, maxt])
    plt.ylim([ymin, ymax])
    plt.legend()
    plt.title("model: {}".format(m))
    plt.savefig(os.path.join(folder, "model{}_margin.png".format(m)))
    if LATEX:
        tikzplotlib.save(os.path.join(folder, "model{}_margin.tex".format(m)))
    plt.close()

ymax = -np.inf
TOSAVE = {}
for alg_name in total_experiments.keys():
    regret = np.array(total_experiments[alg_name])
    rep, T = regret.shape
    t = np.arange(1, T + 1)
    mean_regret = np.mean(regret, axis=0)
    std = np.std(regret, axis=0) / np.sqrt(rep)
    plt.plot(t[::EVERY], mean_regret[::EVERY], linewidth=LW, label=alg_name)
    plt.fill_between(t[::EVERY], mean_regret[::EVERY] - 2 * std[::EVERY], mean_regret[::EVERY] + 2 * std[::EVERY],
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
average_name = os.path.join(folder, "avg_regret.png")
print("Saving average performance to %s ..." % average_name)
plt.savefig(average_name)
average_name = os.path.join(folder, "avg_regret.tex")
tikzplotlib.save(average_name)
print("Done.\n")

avg_margin_name = os.path.join(folder, "avg_margin.png")
print("Saving average margin to %s..." % avg_margin_name)
plt.figure(figsize=(10, 10))
ymax = -np.inf
ymin = np.inf
maxt = -np.inf
TOSAVE = {}
for alg_name in avg_margin.keys():
    margin = np.array(avg_margin[alg_name])
    rep, T = margin.shape
    t = np.arange(1, T + 1)
    mean_margin = np.mean(margin, axis=0)
    std = np.std(margin, axis=0) / np.sqrt(rep)
    idxs = mean_margin < 2
    if np.sum(idxs) > 0:
        plt.plot(t[idxs], mean_margin[idxs], linewidth=LW, label=alg_name)
        plt.fill_between(t[idxs],
                         mean_margin[idxs] - 2 * std[idxs], mean_margin[idxs] + 2 * std[idxs],
                         alpha=0.15)
        ymin = min(ymin, np.min(mean_margin[idxs] - 2 * std[idxs]))
        ymax = max(ymax, np.max(mean_margin[idxs] + 2 * std[idxs]))
        maxt = max(maxt, np.max(t[idxs]))

    M = np.concatenate((t.reshape(-1, 1), mean_margin.reshape(-1, 1), std.reshape(-1, 1)), axis=1)
    TOSAVE[alg_name] = M

plt.xlim([1, maxt])
# plt.ylim([0, ymax])
plt.xlabel("Time")
plt.ylabel("Average Margin")
plt.legend()
plt.savefig(avg_margin_name)
average_name = os.path.join(folder, "avg_margin.tex")
tikzplotlib.save(average_name)
plt.close()

np.savez_compressed(os.path.join(folder, "avg_margin"), **TOSAVE)
print("Done.\n")

print(bad_model, min_val)
plt.figure(figsize=(10, 10))
plt.title("Model: {}".format(bad_model))
ymax = -np.inf
T = None
for model in [results[bad_model]]:  # results:
    print(model[2])
    # for el in model[2]:
    #     print(el.mean)
    for alg_name, val in model[1]:
        print(alg_name)
        rep, T = val['cum_rewards'].shape

        t = np.arange(1, T + 1)
        regret = np.cumsum(val['regret'], axis=1)
        mean_regret = np.mean(regret, axis=0)
        std = np.std(regret, axis=0) / np.sqrt(rep)
        low_quantile = np.quantile(regret, 0.25, axis=0)
        high_quantile = np.quantile(regret, 0.75, axis=0)
        # rwds = np.mean(val['cum_rewards'], axis=0)
        # low_quantile_rwds = np.quantile(val['cum_rewards'], 0.25, axis=0)
        # high_quantile_rwds = np.quantile(val['cum_rewards'], 0.75, axis=0)

        plt.plot(t[::EVERY], mean_regret[::EVERY], linewidth=LW, label=alg_name)
        plt.fill_between(t[::EVERY], mean_regret[::EVERY] - 2 * std[::EVERY], mean_regret[::EVERY] + 2 * std[::EVERY],
                         alpha=0.15)
        ymax = max(ymax, mean_regret[-1] + 2 * std[-1])

plt.xlim([0, T])
plt.ylim([0, ymax])
plt.legend()
plt.savefig(os.path.join(folder, "worst_linear_exp.png"))
print("Done.\n")

worst_name = os.path.join(folder, "worst_linear_exp.tex")
print("Saving worst model to %s..." % worst_name)
tikzplotlib.save(worst_name)
print("Done.\n")

archive_name = "{}.tar.gz".format(folder)
print("Compressing files to %s..." % archive_name)
tardir(folder, archive_name)
print("Done.\n")

plt.show()

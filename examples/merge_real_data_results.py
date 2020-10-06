# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import os
from cycler import cycler
import matplotlib.pyplot as plt
import tikzplotlib

n = 9  # Number of colors
new_colors = [plt.get_cmap('Set1')(1. * i / n) for i in range(n)]
linestyle_cycler = cycler('linestyle', ['-', '--', ':', '-.', '-', '--', ':', '-.', '-'])
plt.rc('axes', prop_cycle=(cycler('color', new_colors) + linestyle_cycler))
plt.rc('lines', linewidth=2)

what = 'regret'
what = 'margin'

EVERY = 200
LW = 2
folders = ["20190905_043752_Bernoulli_PAR_martingale_results"]
plt.figure(figsize=(20, 10))
T = 0
if what == 'margin':
    ymax = np.inf
else:
    ymax = -np.inf
for fname in folders:
    M = np.load(os.path.join(fname, "avg_{}.npz".format(what)), mmap_mode='r')
    for alg in M.files:
        if not alg  in ["CUCB-oracle-0.05", "CUCB-new-0.05-1",
                        "CUCB-LBS-new-0.05-1",
                        "CSUCB-old-0.05-1","CUCB-LBS-old-0.05-1"]:
            data = M[alg]
            t = data[:, 0]
            yval = data[:, 1]
            std = data[:, 2]
            plt.plot(t[::EVERY], yval[::EVERY], linewidth=LW, label=alg)
            plt.fill_between(t[::EVERY],
                             yval[::EVERY] - 2 * std[::EVERY], yval[::EVERY] + 2 * std[::EVERY],
                             alpha=0.15)
            if what == 'margin':
                ymax = min(ymax, np.min(yval - 2 * std))
            else:
                ymax = max(ymax, yval[-1] + 2 * std[-1])
            T = max(T, np.max(t))

plt.plot([0,T], [0, 0], '-', c='gray', linewidth=0.8)
plt.xlim([0, T])
# if ymax > 0:
#    plt.ylim([0, ymax])
# else:
#    plt.ylim([ymax, 5])
plt.xlabel("Time")
if what == "regret":
    plt.ylabel("Cumulative Regret")
else:
    plt.ylabel("Average Budget")
plt.legend()
plt.savefig("jester_average_{}.png".format(what))
tikzplotlib.save("jester_average_{}.tex".format(what))
plt.show()

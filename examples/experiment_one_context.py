# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import datetime
import json
import os
import pickle
import sys
from collections import namedtuple

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

sys.path.append('/private/home/broz/workspaces/bandits_attacks')
from isoexp.contextual.contextual_models import DatasetModel, RandomContextualLinearArms

from examples.linear_contextual_bandit import work
from scipy.optimize import minimize, linprog

"""
TEST Linear Bandit 
"""

def in_hull(points, x):
    n_points = len(points)
    n_dim = len(x)
    c = np.zeros(n_points)
    A = np.r_[points.T, np.ones((1, n_points))]
    b = np.r_[x, np.ones(1)]
    lp = linprog(c, A_eq=A, b_eq=b)
    return lp.success

if __name__ == '__main__':

    PARALLEL = True
    print("PARALLEL: {}".format(PARALLEL))

    MABResults = namedtuple('MABResults', 'regret, cum_rewards, norm_errors')

    random_state = np.random.randint(0, 123123)
    np.random.seed(random_state)
    local_random = np.random.RandomState(random_state)

    print("seed: {}".format(random_state))

    K = 10
    n_features = 30
    nb_arms=10
    a_noise = 0.1

    la = 1. / 2.
    delta = 0.99
    reg_factor = 0.1
    just_a_test= True
    if just_a_test:
        T = 5 * 10 ** 4  # horizon
        nb_simu = 5  # 48 * 5 #240
    else:
        T = int(1 * 10 ** 6) # horizon
        nb_simu = 40  # 48 * 5 #240
    adversarial_xi = 0.0001
    noise=0.1
    attack_frequencies = ['target_arm', 0.0]  # [1.0, 'decrease_sqrt', 0]
    algo_names = ['LinUCB', 'eps_greedy', 'LinTS']
    weak_attacks_list = [False]  # [False, True] #
    methods_to_test = [None]  # ['quadprog', 'heuristic']
    sparse_factors = [None]
    results = []
    decrease_epsilon = True

    movielens = True
    jester = False
    dataset_model = movielens or jester
    assert(not(movielens and jester)), "cannot use both movielens and jester"
    if dataset_model:
        if movielens:
            simulator = DatasetModel(os.path.abspath('examples/movielens/Vt_movielens.csv'), user_csvfile=os.path.abspath("examples/movielens/U.csv"), arms_limit=25, noise=noise, context_limit=100)
        elif jester:
            simulator = DatasetModel(os.path.abspath("examples/jester/Vt_jester.csv"), user_csvfile=os.path.abspath('examples/jester/U.csv'), noise=noise, context_limit=100)
        else:
            print('Issue, should use a dataset that isn\'t jester or movielens')
            exit(0)

    else:
        simulator = RandomContextualLinearArms(n_actions=nb_arms, n_features=n_features, noise=noise, bound_context=1)
        # target_context = np.random.randint(low=0, high=len(simulator.context_lists))
        # x_star = simulator.context_lists[target_context]
        # means_x_star = np.dot(simulator.thetas, x_star)

    target_context = np.random.randint(low=0, high=len(simulator.context_lists))
    x_star = simulator.context_lists[target_context]
    means_x_star = np.dot(simulator.thetas, x_star)

    target_arm = np.argmin(means_x_star)
    method= 'linUCB_Relaxed'
    settings = {
        "T": T,
        'models': algo_names,
        "nb_simu": nb_simu,
        "random_state": random_state,
        "K": simulator.n_actions if simulator else K,
        "dimension": simulator.n_features if simulator else n_features,
        'attack_frequencies': attack_frequencies,
        'weak_attacks': weak_attacks_list,
        'methods_to_test': methods_to_test,
        'adversarial_xi': adversarial_xi,
        'sparse_factors': sparse_factors,
        'target_arm': target_arm,
    }
    weak_attack=False
    dataset_type = 'jester' if jester else 'movilens' if movielens else 'simulation'
    print(f'running on {dataset_type}')
    mask = np.ones(simulator.n_actions, dtype='int')
    mask[target_arm] = 0

    print(in_hull(x=simulator.thetas[target_arm], points=np.array(simulator.thetas[mask])))
    if in_hull(x=simulator.thetas[target_arm], points=np.array(simulator.thetas[mask])):
        raise ValueError()

    if PARALLEL:
        import multiprocessing
        work_to_be_done = []

        for alg_name in algo_names:
            for attack_frequency in attack_frequencies:
                for sparse_attacks in sparse_factors:
                    for sim_index in range(nb_simu):
                                work_to_be_done.append((attack_frequency, False, weak_attack, 'quadprog' if alg_name == 'eps_greedy' else 'linUCB_Relaxed' if alg_name == 'LinUCB' else 'TS_Relaxed' if alg_name=='LinTS' else None, adversarial_xi, sim_index, alg_name, x_star))

        # for sim_index in range(nb_simu):
        # #     work_to_be_done.append((0.2, 10, False, 'quadprog', xi, sim_index))
        #     work_to_be_done.append((0.2, 10, False, 'quadprog', adversarial_xi[0], sim_index))
        settings['work_list'] = work_to_be_done
        num_cores = multiprocessing.cpu_count()
        results.append(Parallel(n_jobs=num_cores, verbose=1)(
            delayed(work)(nb_arms=K, noise=a_noise, n_features=n_features, T=T, random_state=random_state + sim_index,
                           attack_frequency=attack_frequency,alg_name=alg_name,
                          weak_attack=weak_attack, adversarial_xi=xi, method=method,
                          sparse_attacks=sparse_attacks, simulator=simulator, target_arm=target_arm, x_star=x_star) for
            attack_frequency, sparse_attacks, weak_attack, method, xi, sim_index, alg_name, x_star in work_to_be_done))
    else:
        #     for decrease_epsilon in [True, False]:
        for attack_frequency in [0]:  # [1.0,0.1, 0]:
            weak_attack = False
            for k in tqdm(range(nb_simu)):
                ret = work(nb_arms=K, noise=a_noise, n_features=n_features, T=T, random_state=random_state + k,
                           attack_frequency=attack_frequency,
                           weak_attack=weak_attack)
                results.append(ret)

    id = '{}_{:%Y%m%d_%H%M%S}_{}_alg{}{}'.format(dataset_type, datetime.datetime.now(), 'linear_one_context', algo_names, '_test' if just_a_test else '')
    pickle_name = "{}_{}_linear_results.pickle".format(id, "PAR" if PARALLEL else "SEQ")
    print(pickle_name)
    with open(pickle_name, "wb") as f:
        pickle.dump(results, f)
    with open("{}_{}_linear_settings.json".format(id, "PAR" if PARALLEL else "SEQ"), "w+") as f:
        json.dump(settings, f)

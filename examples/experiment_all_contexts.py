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
"""
TEST Linear Bandit 
"""



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
    just_a_test = False
    if just_a_test:
        T = 1 * 10 ** 4  # horizon
        nb_simu = 5  # 48 * 5 #240
    else:
        T = 1 * 10 ** 6  # horizon
        nb_simu = 20  # 48 * 5 #240
    adversarial_xi = 0.0001
    noise=0.1
    attack_frequencies = [1.0, 0.2, 0.0]  # [1.0, 'decrease_sqrt', 0]
    algo_names = ['LinUCB', 'eps_greedy', 'LinTS']
    weak_attacks_list = [False]  # [False, True] #
    methods_to_test = [None]  # ['quadprog', 'heuristic']
    sparse_factors = [2.0]
    results = []
    decrease_epsilon = True
    seed = 1
    movielens = False
    jester = False
    dataset_model = movielens or jester
    assert(not(movielens and jester)), "cannot use both movielens and jester"
    if dataset_model:
        if movielens:
            simulator = DatasetModel(os.path.abspath('examples/movielens/Vt_movielens.csv'), user_csvfile=os.path.abspath("examples/movielens/U.csv"), arms_limit=25, noise=noise)
        elif jester:
            simulator = DatasetModel(os.path.abspath("examples/jester/Vt_jester.csv"), user_csvfile=os.path.abspath('examples/jester/U.csv'), noise=noise)
        else:
            print('Issue, should use a dataset that isn\'t jester or movielens')
            exit(0)

    else:
        simulator = RandomContextualLinearArms(n_actions=nb_arms, n_features=n_features, noise=noise, random_state=seed, bound_context=1)
        # target_context = np.random.randint(low=0, high=len(simulator.context_lists))
        # x_star = simulator.context_lists[target_context]
        # means_x_star = np.dot(simulator.thetas, x_star)

    # target_context = np.random.randint(low=0, high=len(simulator.context_lists))
    # x_star = simulator.context_lists[target_context]
    means_x_star = np.dot(simulator.context_lists, simulator.thetas.T).mean(axis=0)
    target_arm = np.argmin(means_x_star)

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
    method=None
    dataset_type = 'jester' if jester else 'movilens' if movielens else 'simulation'
    print(f'running on {dataset_type}')
    if PARALLEL:
        import multiprocessing
        work_to_be_done = []

        for alg_name in algo_names:
            for attack_frequency in attack_frequencies:
                for sparse_attacks in sparse_factors:
                    for sim_index in range(nb_simu):
                                work_to_be_done.append((attack_frequency, sparse_attacks/ attack_frequency if attack_frequency > 0 else 0, weak_attack, method, adversarial_xi, sim_index, alg_name))

        # for sim_index in range(nb_simu):
        # #     work_to_be_done.append((0.2, 10, False, 'quadprog', xi, sim_index))
        #     work_to_be_done.append((0.2, 10, False, 'quadprog', adversarial_xi[0], sim_index))
        settings['work_list'] = work_to_be_done
        num_cores = multiprocessing.cpu_count()
        results.append(Parallel(n_jobs=num_cores, verbose=1)(
            delayed(work)(nb_arms=K, noise=a_noise, n_features=n_features, T=T, random_state=random_state + sim_index,
                           attack_frequency=attack_frequency,alg_name=alg_name,
                          weak_attack=weak_attack, adversarial_xi=xi, method=method,
                          sparse_attacks=sparse_attacks, simulator=simulator, target_arm=target_arm) for
            attack_frequency, sparse_attacks, weak_attack, method, xi, sim_index, alg_name in work_to_be_done))
    else:
        #     for decrease_epsilon in [True, False]:
        for attack_frequency in [0]:  # [1.0,0.1, 0]:
            weak_attack = False
            for k in tqdm(range(nb_simu)):
                ret = work(nb_arms=K, noise=a_noise, n_features=n_features, T=T, random_state=random_state + k,
                           attack_frequency=attack_frequency,
                           weak_attack=weak_attack)
                results.append(ret)

    id = '{}_{:%Y%m%d_%H%M%S}_{}_alg{}'.format(dataset_type, datetime.datetime.now(), '_Contextual_linear_all_contextes', algo_names)
    pickle_name = "{}_{}_linear_results.pickle".format(id, "PAR" if PARALLEL else "SEQ")
    print(pickle_name)
    with open(pickle_name, "wb") as f:
        pickle.dump(results, f)
    with open("{}_{}_linear_settings.json".format(id, "PAR" if PARALLEL else "SEQ"), "w+") as f:
        json.dump(settings, f)

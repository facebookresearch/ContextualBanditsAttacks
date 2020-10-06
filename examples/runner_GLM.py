# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import sys
sys.path[0]  = '/Users/evrard/Desktop/monotone_mabs/'
import numpy as np
import isoexp.linear.linearmab_models as arms
import isoexp.linear.linearbandit as mabs
import isoexp.conservative.linearmabs as cmabs
from tqdm import tqdm, trange
from joblib import Parallel, delayed
import math
import dill
import json
import datetime

from collections import namedtuple

"""
TEST GLM Bandit 
Compare between martingale and sum of Hoffding bounds
"""


def work(m, nb_arms, nb_features, noise, b_pos, nb_simu, T, all_algs, random_state):
    # create model
    K = nb_arms
    model = arms.RandomLogArms(n_actions = K, 
                                  n_features = n_features, 
                                  random_state = random_state + m, 
                                  bound_features = 1, 
                                  bound_theta = 1,
                                  noise = noise)
    means = model.link(np.dot(model.features,model.theta))
    kappa = model.kappa
    theta_bound = np.linalg.norm(model.theta, 2)

    # Define baseline
    baseline = np.argsort(means)[b_pos]
    mean_baseline = means[baseline]

    AAA = []
    for alg_name in tqdm(all_algs.keys(), desc='Sim. model {}'.format(m)):
        
        alg = all_algs[alg_name](model.features, noise, theta_bound, 
                      mean_baseline, baseline, kappa = kappa)
        regret = np.zeros((nb_simu, T))
        rwds = regret.copy()
        norms = regret.copy()
        
        for k in trange(nb_simu, desc = 'Repetitions'):
            
            alg.reset()
            
            for t in trange(T, desc = 'Inside episode') :
                            
                a_t = alg.get_action()
                r_t = model.reward(a_t)
                if hasattr(alg, 'conservative_level'):
                    rwds[k,t] = means[a_t] - (1 - alg.conservative_level)*mean_baseline
                else :
                    rwds[k,t] = means[a_t]
                alg.update(a_t, r_t)
                regret[k, t] = model.best_arm_reward() - means[a_t]
                if hasattr(alg, 'theta_hat'):
                    norms[k, t] = np.linalg.norm(alg.theta_hat - model.theta, 2)
        
        AAA += [(alg_name, {"regret": regret, "cum_rewards": rwds.cumsum(axis = 1), "norm_errors" : norms})]

    return m, AAA, model, mean_baseline


# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    
    PARALLEL = True
    print("PARALLEL: {}".format(PARALLEL))

    MABResults = namedtuple('MABResults', 'regret, cum_rewards, norm_errors')

    random_state = np.random.randint(0, 123123)
    np.random.seed(random_state)
    print("seed: {}".format(random_state))

    K = 20
    n_features = 10
    a_noise = 0.1
    
    delta = 0.01
    la = 1/4

    T = 3000 # horizon
    nb_models = 4
    nb_simu = int(np.sqrt(T))

    CLEVELS = [0.1]
    BATCHES = [1]
    pos = 15

    settings = {
        "T": T,
        "nb_simu": nb_simu,
        "random_state": random_state,
        "K": K,
        "dimension" : n_features,
        "baseline": pos,
        "conservative_levels": CLEVELS,
        "batches": BATCHES
    }

    algorithms = {
        'GLM-UCB': lambda feat, noise, b_theta, mean_b = 0, b = 0, alpha = 0, kappa = 1 : mabs.UCB_GLM(
                reg_factor=la,
                delta=delta,
                arm_features = feat,
                noise_variance = noise,
                bound_theta = b_theta,
                kappa = kappa,
                model = 'bernoulli',
                tighter_ucb = True)
    }

    conservative_algorithms = {}

    for conservative_level in CLEVELS:
                conservative_algorithms.update(
                    {
                        "GLM-CUCB-{}".format(conservative_level):
                            lambda feat, noise, b_theta, mean_b, b, alpha = conservative_level, kappa = 1: 
                                cmabs.CUCB_GLM(arm_features = feat, 
                                                noise_variance = noise,
                                                bound_theta = b_theta,
                                                mean_baseline = mean_b,
                                                baseline = b,
                                                reg_factor = la,
                                                delta = delta,
                                                conservative_level = alpha,
                                                kappa = kappa,
                                                tighter_ucb = True,
                                                model = 'bernoulli'),                    }
                )

    results = []
    full_algo = {**algorithms, **conservative_algorithms}
    if PARALLEL:
        import multiprocessing

        num_cores = multiprocessing.cpu_count()
        results = Parallel(n_jobs=num_cores, verbose=1)(
            delayed(work)(m=m, nb_arms=K, nb_features = n_features, noise = a_noise, b_pos=pos,
                          nb_simu=nb_simu, T=T, all_algs=full_algo,
                          random_state=random_state) for m in range(nb_models))

    else:

        for m in tqdm(range(nb_models)):
            ret = work(m, K, n_features, a_noise, pos, nb_simu, T, full_algo, random_state)
            results.append(ret)
            # MAB = []
            # means = None
            # if ARMS == "Bernoulli":
            #     means = np.random.uniform(low=0.25, high=0.75, size=K)
            #     for n in range(K):
            #         MAB.append(arms.ArmBernoulli(p=means[n], random_state=random_state + n))
            # elif ARMS == "TruncatedNormal":
            #     means = np.random.uniform(low=0., high=1., size=K)
            #     sigmas = np.random.uniform(low=0.1, high=1., size=K)
            #     for n in range(K):
            #         MAB.append(arms.ArmTruncNorm(original_mean=means[n], a=0, b=1, original_std=sigmas[n]))
            #         means[n] = MAB[n].mean
            #         sigmas[n] = MAB[n].sigma
            # else:
            #     raise ValueError("Unknown arm type")
            #
            # AAA = []
            # for alg_name in full_algo.keys():
            #     alg = full_algo[alg_name]
            #
            #     regret = np.zeros((nb_simu, T))
            #     rwds = 0 * regret
            #
            #     for k in range(nb_simu):
            #         # Define baseline
            #         baseline = np.argsort(means)[pos]
            #         mean_baseline = MAB[baseline].mean
            #
            #         rewards, draws = alg(T, MAB, baseline, mean_baseline)
            #         regret[k] = max(means) * np.arange(1, T + 1) - np.cumsum(rewards)
            #         rwds[k] = np.cumsum(means[draws.astype('int')])
            #
            #     AAA += [(alg_name, {"regret": regret, "cum_rewards": rwds})]
            #
            # results += [(m, AAA, MAB)]
    results += [CLEVELS]
    id = '{:%Y%m%d_%H%M%S}_{}'.format(datetime.datetime.now(), 'GLM')
    with open("{}_{}_GLM_results.pickle".format(id, "PAR" if PARALLEL else "SEQ"), "wb") as f:
        dill.dump(results, f)
    with open("{}_{}_GLM_settings.json".format(id, "PAR" if PARALLEL else "SEQ"), "w+") as f:
        json.dump(settings, f)

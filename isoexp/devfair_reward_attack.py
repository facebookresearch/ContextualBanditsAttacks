# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import sys
sys.path.append('/private/home/broz/workspaces/bandits_attacks')

import  isoexp.contextual.contextual_models as arms
from isoexp.contextual.contextual_linucb import *
from tqdm import tqdm
from cycler import cycler
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import math
import pickle
from tqdm import trange
import json
import datetime
from collections import namedtuple
import re
import os

frequency = 100
class exp(object):
    def __init__(self, nb_arms, type='random', a_star=0, m=None):
        self.K = nb_arms
        self.type = type
        self.a_star = a_star
        self.m = m

    def get_action(self, context):
        if self.type == 'random':
            return np.ones((self.K,)) / self.K
        elif self.type == 'optimal':
            means = np.dot(self.m.thetas, context)
            a = np.argmax(means)
            proba = np.zeros((self.K,))
            proba[a] = 1
            return proba
        else:
            proba = np.zeros((self.K,))
            proba[self.a_star] = 1
            return proba


def work(m, nb_arms, nb_features, noise, nb_simu, T, all_algs, random_state, M=1, bound_context=1, dataset=False, which=None):
    # create model
    K = nb_arms
    if dataset:
        if which == 'jester':
            arm_file = os.path.abspath("examples/jester/Vt_jester.csv")
            user_file = os.path.abspath("examples/jester/U.csv")
            model = arms.DatasetModel(arm_csvfile=arm_file, user_csvfile=user_file, noise=noise, random_state=random_state)
        else:
            arm_file = os.path.abspath('examples/movielens/Vt_movielens.csv')
            user_file = os.path.abspath('examples/movielens/U.csv')
            model = arms.DatasetModel(arm_csvfile=arm_file, user_csvfile=user_file, noise=noise, random_state=random_state,  arms_limit=25)
    else:
        model = arms.RandomContextualLinearArms(n_actions=K, n_features=nb_features, noise=noise,
                                                random_state=random_state, bound_context=bound_context)
    theta_bound = np.max(np.linalg.norm(model.thetas, axis=1))
    target_context = np.random.randint(low=0, high=len(model.context_lists))
    other_context = np.random.randint(low=0, high=len(model.context_lists))
    # while other_context == target_context:
    #     other_context = np.random.randint(low=0, high=len(model.context_lists))
    target_arm = np.random.randint(low=0, high=model.n_actions)
    AAA = []
    for alg_name in tqdm(all_algs.keys(), desc='Sim. model {}'.format(m)):
        args = {'nb_arms': model.n_actions,
                'dimension': model.n_features,
                'bound_features': theta_bound,
                'bound_context': model.bound_context,
                'reg_factor': 0.1,
                'delta': delta,
                'noise_variance': noise,
                }
        if 'Exp4' in alg_name:
            eta = np.sqrt(2 * np.log(M) / (T * model.n_actions))
            experts = []
            for i in range(M - 2):
                experts.append(exp(nb_arms=model.n_actions, type='random'))
            experts.append(exp(nb_arms=model.n_actions, type='optimal', m=model))
            experts.append(exp(nb_arms=model.n_actions, type='', a_star=int(target_arm)))
            args['experts'] = experts
            args['eta'] = eta
        alg = all_algs[alg_name](**args)
        if 'attacked' in alg_name:
            if 'gamma' in alg_name:
                temp_eps = re.findall(r'[\d\.\d]+', alg_name)
                temp_eps = np.array(list(map(float, temp_eps)))
                temp_eps = temp_eps[temp_eps<=1]
                temp_eps = temp_eps[0]
                temp_args = args.copy()
                temp_args['eps'] = temp_eps
                attacker = RewardAttacker(**temp_args)
        regret = np.zeros((nb_simu, T//frequency))  #[[]] * nb_simu #np.zeros((nb_simu, T))
        draws = regret.copy()
        epsilon_norm = np.zeros((nb_simu, T//frequency))  #[[]] * nb_simu #np.zeros((nb_simu, T))
        # thetas_alg = np.zeros((nb_simu, T, model.n_actions, model.n_features))
        # prod_scalar = np.zeros((nb_simu, T, model.n_actions, model.n))
        rewards_range =  np.zeros((nb_simu, T//frequency))  #[[]] * nb_simu # np.zeros((nb_simu, T))

        for k in range(nb_simu):

            alg.reset()

            if 'attacked' in alg_name and not 'stationary' in alg_name:
                attacker.reset()

            attack_acumulator = 0
            regret_accumulator = 0
            rewards_range_max = 0
            draws_accumulator = 0
            for t in trange(T):

                context = model.get_context()
                a_t = alg.get_action(context)
                r_t = model.reward(context, a_t)
                if 'attacked' in alg_name:
                    if not 'stationary' in alg_name:
                        attacker.update(context, a_t, r_t)
                        attack_t = attacker.compute_attack(a_t, context, target_arm)
                    else:
                        if a_t != target_arm:
                            attack_t = -r_t + noise*np.random.randn()
                        else:
                            attack_t = 0
                    # print('attack_t =', attack_t)
                else:
                    attack_t = 0
                alg.update(context, a_t, min(1, max(0, r_t+attack_t)))

                attack_acumulator+= np.abs(attack_t)
                regret_accumulator+= model.best_arm_reward(context) - np.dot(model.thetas[a_t], context)
                rewards_range_max = max(rewards_range_max, min(1, max(r_t + attack_t, 0)))
                draws_accumulator +=1 if a_t == target_arm else 0
                if t % frequency == 0: # logging
                    epsilon_norm[k, t // frequency]= attack_acumulator
                    regret[k, t // frequency]= regret_accumulator
                    rewards_range[k, t // frequency]= rewards_range_max
                    draws[k, t // frequency]= draws_accumulator
                    attack_acumulator = 0
                    regret_accumulator = 0
                    rewards_range_max = 0
                    draws_accumulator = 0

                # print('reward = ', min(1, max(r_t + attack_t, 0)))
                # print('Target arm =', target_arm, 'a_t =', a_t)
                # alg.update(context, a_t, r_t + attack_t)
                # if hasattr(alg, 'thetas_hat'):
                    # thetas_alg[k, t] = alg.thetas_hat
                    # for a in range(model.n_actions):
                    #     for i, x in enumerate(model.context_lists):
                    #         if 'attacked' in alg_name:
                    #             p = np.dot(alg.thetas_hat[a], x) - (1 - attacker.eps) * np.dot(model.thetas[target_arm], x)
                    #         else:
                    #             p = np.dot(alg.thetas_hat[a], x) - np.dot(model.thetas[target_arm], x)
                    #         prod_scalar[k, t, a, i] = p
                # print('-'*100)
                # print('r_t =', r_t)
                # print('atttack_t =', attack_t)
                # print('r_t + attack_t = ', r_t + attack_t)
                # rewards_range[k, t] = min(1, max(r_t + attack_t, 0))




        AAA += [(alg_name, {"regret": regret, "attack_cond": epsilon_norm, "target_draws": draws, "thetas": (),
                            "prod_scalar": (), "range_rewards": rewards_range})]

    return m, AAA, model, target_arm


def run_and_output(dataset=None):
    results = []
    if PARALLEL:
        import multiprocessing

        num_cores = multiprocessing.cpu_count()
        results = Parallel(n_jobs=num_cores, verbose=1)(
            delayed(work)(m=m, nb_arms=K, nb_features=n_features, noise=a_noise,
                          nb_simu=nb_simu, T=T, all_algs=algorithms,
                          random_state=random_state + m, M=M, which=dataset) for m in range(nb_models))

    else:

        for m in tqdm(range(nb_models)):
            ret = work(m, K, n_features, a_noise, nb_simu, T, algorithms, random_state + m, M=M)
            results.append(ret)
    id = '{:%Y%m%d_%H%M%S}'.format(datetime.datetime.now())
    pickle_name = "{}_{}_{}_contextual_attacks_rewards.pickle".format(dataset, id, "PAR" if PARALLEL else "SEQ")
    print(pickle_name)
    with open(pickle_name, "wb") as f:
        pickle.dump(results, f)
    with open("{}_{}_{}_contextual_attacks_rewards_settings.json".format(dataset, id, "PAR" if PARALLEL else "SEQ"), "w+") as f:
        json.dump(settings, f)
    return results, pickle_name, id,


if __name__ == '__main__':

    PARALLEL = False
    print("PARALLEL: {}".format(PARALLEL))

    MABResults = namedtuple('MABResults', 'regret, attack_cond, target_draws, thetas, prod_scalar')

    random_state = np.random.randint(0, 123123)
    np.random.seed(random_state)
    print("seed: {}".format(random_state))

    K = 10
    n_features = 30
    a_noise = 0.1

    delta = 0.01
    la = 0.1


    T = 1*10**6  # horizon
    nb_models = 5
    nb_simu = 25
    M = 5
    # attack_parameter_to_test = np.linspace(0, 1, 10)
    attack_parameter_to_test = np.array([1/2])
    settings = {
        "T": T,
        "nb_simu": nb_simu,
        "nb_models": nb_models,
        "random_state": random_state,
        "K": K,
        "dimension": n_features,
        "epsilon_tested": list(attack_parameter_to_test),
        'frequency': frequency

    }

    algorithms = {
        'LinUCB': lambda nb_arms, dimension, bound_features, bound_context, reg_factor=la, delta=delta, noise_variance=a_noise:
            ContextualLinearBandit(reg_factor=la,

        #
        # 'LinTS': lambda nb_arms, dimension, bound_features, bound_context, reg_factor=la, delta=delta,
        #                 noise_variance=a_noise: ContextualLinearTS(nb_arms=nb_arms,
        #                                                            dimension=dimension,
        #                                                            reg_factor=reg_factor,
        #                                                            delta=delta,
        #                                                            noise_variance=noise_variance),

        # 'Exp4': lambda nb_arms, dimension, experts, eta, bound_features, bound_context, reg_factor=la, delta=delta,
        #                 noise_variance=a_noise: Exp4(nb_arms=nb_arms,
        #                                              dimension=dimension,
        #                                              experts=experts,
        #                                              eta=eta,
        #                                              gamma=0),

        # 'eps-greedy': lambda nb_arms, dimension, bound_features, bound_context, reg_factor=la, delta=delta,
        #                 noise_variance=a_noise: contextEpsGREEDY(number_arms=nb_arms, dimension=dimension,
        #                                                           decrease_epsilon=True, reg_factor=reg_factor),
                delta=delta,
                nb_arms=nb_arms,
                dimension=dimension,
                noise_variance=noise_variance,
                bound_features=bound_features,
                bound_context=bound_context),
        'LinTS': lambda nb_arms, dimension, bound_features, bound_context, reg_factor=la, delta=delta,
                        noise_variance=a_noise: ContextualLinearTS(nb_arms=nb_arms,
                                                                   dimension=dimension,
                                                                   reg_factor=reg_factor,
                                                                   delta=delta,
                                                                   noise_variance=noise_variance),
        'Exp4': lambda nb_arms, dimension, experts, eta, bound_features, bound_context, reg_factor=la, delta=delta,
                        noise_variance=a_noise: Exp4(nb_arms=nb_arms,
                                                     dimension=dimension,
                                                     experts=experts,
                                                     eta=eta,
                                                     gamma=0),
        'eps-greedy': lambda nb_arms, dimension, bound_features, bound_context, reg_factor=la, delta=delta,
                        noise_variance=a_noise: contextEpsGREEDY(number_arms=nb_arms, dimension=dimension,
                                                                  decrease_epsilon=True, reg_factor=reg_factor),

    }

    algorithms.update({

        # 'LinUCB attacked stationary': lambda nb_arms, dimension, bound_features, bound_context, reg_factor=la, delta=delta, noise_variance=a_noise:
        #     ContextualLinearBandit(reg_factor=la,
        #                            delta=delta,
        #                            nb_arms=nb_arms,
        #                            dimension=dimension,
        #                            noise_variance=noise_variance,
        #                            bound_features=bound_features,
        #                            bound_context=bound_context),
        #
        # 'LinTS attacked stationary': lambda nb_arms, dimension, bound_features, bound_context, reg_factor=la, delta=delta,
        #                 noise_variance=a_noise: ContextualLinearTS(nb_arms=nb_arms,
        #                                                            dimension=dimension,
        #                                                            reg_factor=reg_factor,
        #                                                            delta=delta,
        #                                                            noise_variance=noise_variance),

        # 'Exp4 attacked stationary': lambda nb_arms, dimension, experts, eta, bound_features, bound_context, reg_factor=la, delta=delta,
        #                 noise_variance=a_noise: Exp4(nb_arms=nb_arms,
        #                                              dimension=dimension,
        #                                              experts=experts,
        #                                              eta=eta,
        #                                              gamma=0),
        #
        # 'eps-greedy attacked stationary': lambda nb_arms, dimension, bound_features, bound_context, reg_factor=la, delta=delta,
        #                 noise_variance=a_noise: contextEpsGREEDY(number_arms=nb_arms, dimension=dimension,
        #                                                           decrease_epsilon=True, reg_factor=reg_factor),

        'LinUCB attacked stationary': lambda nb_arms, dimension, bound_features, bound_context, reg_factor=la, delta=delta, noise_variance=a_noise:
            ContextualLinearBandit(reg_factor=la,
                                   delta=delta,
                                   nb_arms=nb_arms,
                                   dimension=dimension,
                                   noise_variance=noise_variance,
                                   bound_features=bound_features,
                                   bound_context=bound_context),

        'LinTS attacked stationary': lambda nb_arms, dimension, bound_features, bound_context, reg_factor=la, delta=delta,
                        noise_variance=a_noise: ContextualLinearTS(nb_arms=nb_arms,
                                                                   dimension=dimension,
                                                                   reg_factor=reg_factor,
                                                                   delta=delta,
                                                                   noise_variance=noise_variance),

        'Exp4 attacked stationary': lambda nb_arms, dimension, experts, eta, bound_features, bound_context, reg_factor=la, delta=delta,
                        noise_variance=a_noise: Exp4(nb_arms=nb_arms,
                                                     dimension=dimension,
                                                     experts=experts,
                                                     eta=eta,
                                                     gamma=0),

        'eps-greedy attacked stationary': lambda nb_arms, dimension, bound_features, bound_context, reg_factor=la, delta=delta,
                        noise_variance=a_noise: contextEpsGREEDY(number_arms=nb_arms, dimension=dimension,
                                                                  decrease_epsilon=True, reg_factor=reg_factor),

    })

    for eps in attack_parameter_to_test:
        algorithms.update({
            # 'LinUCB attacked gamma {}'.format(eps): lambda nb_arms, dimension, bound_features, bound_context,
            #                                               reg_factor=la, delta=delta, noise_variance=a_noise:
            #                                           ContextualLinearBandit(reg_factor=la,
            #                                                                  delta=delta,
            #                                                                  nb_arms=nb_arms,
            #                                                                  dimension=dimension,
            #                                                                  noise_variance=noise_variance,
            #                                                                  bound_features=bound_features,
            #                                                                  bound_context=bound_context),

            # 'LinTS attacked gamma {}'.format(eps): lambda nb_arms, dimension, bound_features, bound_context, reg_factor=la, delta=delta,
            #                          noise_variance=a_noise: ContextualLinearTS(nb_arms=nb_arms,
            #                                                                     dimension=dimension,
            #                                                                     reg_factor=reg_factor,
            #                                                                     delta=delta,
            #                                                                     noise_variance=noise_variance),

            # 'Exp4 attacked gamma {}'.format(eps): lambda nb_arms, dimension, experts, eta, bound_features, bound_context, reg_factor=la, delta=delta,
            #                         noise_variance=a_noise: Exp4(nb_arms=nb_arms,
            #                                                      dimension=dimension,
            #                                                      experts=experts,
            #                                                      eta=eta,
            #                                                      gamma=0),
            #
            # 'eps-greedy attacked gamma {}'.format(eps): lambda nb_arms, dimension, bound_features, bound_context, reg_factor=la, delta=delta,
            #                         noise_variance=a_noise: contextEpsGREEDY(number_arms=nb_arms, dimension=dimension,
            #                                                       decrease_epsilon=True, reg_factor=reg_factor)

            'LinUCB attacked gamma {}'.format(eps): lambda nb_arms, dimension, bound_features, bound_context,
                                                          reg_factor=la, delta=delta, noise_variance=a_noise:
                                                      ContextualLinearBandit(reg_factor=la,
                                                                             delta=delta,
                                                                             nb_arms=nb_arms,
                                                                             dimension=dimension,
                                                                             noise_variance=noise_variance,
                                                                             bound_features=bound_features,
                                                                             bound_context=bound_context),

            'LinTS attacked gamma {}'.format(eps): lambda nb_arms, dimension, bound_features, bound_context, reg_factor=la, delta=delta,
                                     noise_variance=a_noise: ContextualLinearTS(nb_arms=nb_arms,
                                                                                dimension=dimension,
                                                                                reg_factor=reg_factor,
                                                                                delta=delta,
                                                                                noise_variance=noise_variance/5),

            'Exp4 attacked gamma {}'.format(eps): lambda nb_arms, dimension, experts, eta, bound_features, bound_context, reg_factor=la, delta=delta,
                                    noise_variance=a_noise: Exp4(nb_arms=nb_arms,
                                                                 dimension=dimension,
                                                                 experts=experts,
                                                                 eta=eta,
                                                                 gamma=0),

            'eps-greedy attacked gamma {}'.format(eps): lambda nb_arms, dimension, bound_features, bound_context, reg_factor=la, delta=delta,
                                    noise_variance=a_noise: contextEpsGREEDY(number_arms=nb_arms, dimension=dimension,
                                                                  decrease_epsilon=True, reg_factor=reg_factor)
        })
    print(algorithms)
    # results, pickle_name, id = run_and_output(dataset=None)
    # results, pickle_name, id = run_and_output(dataset='jester')
    results, pickle_name, id = run_and_output(dataset='movielens')

    else:

        for m in tqdm(range(nb_models)):
            ret = work(m, K, n_features, a_noise, nb_simu, T, algorithms, random_state + m, M=M, dataset=true_data, which=which)
            results.append(ret)

    # id = '{:%Y%m%d_%H%M%S}'.format(datetime.datetime.now())
    # pickle_name = "{}_{}_contextual_attacks_rewards.pickle".format(id, "PAR" if PARALLEL else "SEQ")
    # print(pickle_name)
    # with open(pickle_name, "wb") as f:
    #     pickle.dump(results, f)
    # with open("{}_{}_contextual_attacks_rewards_settings.json".format(id, "PAR" if PARALLEL else "SEQ"), "w+") as f:
    #     json.dump(settings, f)

    n = 9  # Number of colors
    new_colors = [plt.get_cmap('Set1')(1. * i / n) for i in range(n)]
    linestyle_cycler = cycler('linestyle', ['-', '--', ':', '-.', '-', '--', ':', '-.', '-'])
    plt.rc('axes', prop_cycle=(cycler('color', new_colors) + linestyle_cycler))
    plt.rc('lines', linewidth=2)
    for alg_name, res in results[0][1]:
        algorithms[alg_name] = {'regret': np.zeros((nb_models, nb_simu, T)),
                                'cost': np.zeros((nb_models, nb_simu, T))}
    for m in range(nb_models):
        res = results[m][1]
        for i, val in enumerate(res):
            alg_name = val[0]
            val = val[1]
            algorithms[alg_name]['regret'][m, :, :] = val['regret']
            algorithms[alg_name]['cost'][m, :, :] = val['attack_cond']
    plt.figure(1, figsize=(8, 8))
    t = np.linspace(0, T-1, T, dtype='int')
    for alg_name, res in algorithms.items():
        res['regret'] = res['regret'].cumsum(axis=2)
        mean_regret = np.mean(res['regret'], axis=(0, 1))
        low_quantile = np.quantile(res['regret'], 0.1, axis=(0, 1))
        high_quantile = np.quantile(res['regret'], 1 - 0.1, axis=(0, 1))
        plt.plot(mean_regret, label=alg_name)
        # plt.fill_between(t, low_quantile, high_quantile, alpha=0.15)
        plt.title('Cumulative regret')
    plt.legend()
    plt.show()

    # n = 9  # Number of colors
    # new_colors = [plt.get_cmap('Set1')(1. * i / n) for i in range(n)]
    # linestyle_cycler = cycler('linestyle', ['-', '--', ':', '-.', '-', '--', ':', '-.', '-'])
    # plt.rc('axes', prop_cycle=(cycler('color', new_colors) + linestyle_cycler))
    # plt.rc('lines', linewidth=2)
    # for alg_name, res in results[0][1]:
    #     algorithms[alg_name] = {'regret': np.zeros((nb_models, nb_simu, T)),
    #                             'cost': np.zeros((nb_models, nb_simu, T))}
    # for m in range(nb_models):
    #     res = results[m][1]
    #     for i, val in enumerate(res):
    #         alg_name = val[0]
    #         val = val[1]
    #         algorithms[alg_name]['regret'][m, :, :] = val['regret']
    #         algorithms[alg_name]['cost'][m, :, :] = val['attack_cond']
    # plt.figure(1, figsize=(8, 8))
    # t = np.linspace(0, T-1, T, dtype='int')
    # for alg_name, res in algorithms.items():
    #     res['regret'] = res['regret'].cumsum(axis=2)
    #     mean_regret = np.mean(res['regret'], axis=(0, 1))
    #     low_quantile = np.quantile(res['regret'], 0.1, axis=(0, 1))
    #     high_quantile = np.quantile(res['regret'], 1 - 0.1, axis=(0, 1))
    #     plt.plot(mean_regret, label=alg_name)
    #     plt.fill_between(t, low_quantile, high_quantile, alpha=0.15)
    #     plt.title('Cumulative regret')
    #
    # plt.legend()
    #
    # plt.figure(2, figsize=(8,8))
    # t = np.linspace(0, T-1, T, dtype='int')
    # for alg_name, res in algorithms.items():
    #     res['cost'] = res['cost'].cumsum(axis=2)
    #     mean_regret = np.mean(res['cost'], axis=(0, 1))
    #     low_quantile = np.quantile(res['cost'], 0.1, axis=(0, 1))
    #     high_quantile = np.quantile(res['cost'], 1 - 0.1, axis=(0, 1))
    #     plt.plot(mean_regret, label=alg_name)
    #     plt.fill_between(t, low_quantile, high_quantile, alpha=0.15)
    #     plt.title('Total cost')
    #
    # plt.legend()
    #
    # plt.show()
    # for res in results:
    #     alg_name, val = res[1][0], res[1][1]
    #     print(alg_name)
    #     mean_regret = np.mean(val.regret.cumsum(axis=1), axis=0)
    #     t = np.linspace(0, T, T, dtype='int')
    #     low_quantile = np.quantile(val.regret.cumsum(axis=1), 0.1, axis=0)
    #     high_quantile = np.quantile(val.regret.cumsum(axis=1), 0.9, axis=0)
    #
    #     plt.figure(0)
    #     plt.title('Cumulative Regret')
    #     plt.plot(mean_regret, label=alg_name)
    #     plt.fill_between(t, low_quantile, high_quantile, alpha=0.15)
    #     plt.legend()

        # mean_condition = np.mean(np.cumsum(val.target_draws == target_arm, axis=1), axis=0)
        # low_quantile = np.quantile(val.attack_cond, 0.1, axis=0)
        # high_quantile = np.quantile(val.attack_cond, 0.9, axis=0)
        # plt.figure(1)
        # plt.title('Draws target arm')
        # plt.plot(mean_condition, label=alg_name)
        # plt.fill_between(t, low_quantile, high_quantile, alpha=0.15)
        # plt.legend()

        # plt.figure(2)
        # plt.title('Cumulative attack norm attacked reward')
        # if 'Attacked' in alg_name:
        #     plt.plot(np.mean(np.cumsum(val.attack_cond, axis=1), axis=0), label=alg_name)
        #     low_quantile = np.quantile(np.cumsum(val.attack_cond, axis=1), 0.1, axis=0)
        #     high_quantile = np.quantile(np.cumsum(val.attack_cond, axis=1), 0.9, axis=0)
        #     plt.fill_between(t, low_quantile, high_quantile, alpha=0.15)
        # plt.legend()
        #
        #
        # plt.figure(4)
        # plt.title('Error true env and learned env')
        # for a in range(model.n_actions):
        #     error = np.linalg.norm(val.thetas[:, :, a] - model.thetas[a], axis=2)
        #     plt.plot(t, np.mean(error, axis=0), label=alg_name + ' arm {}'.format(a))
        #     low_quantile = np.quantile(error, 0.1, axis=0)
        #     high_quantile = np.quantile(error, 0.9, axis=0)
        #     plt.fill_between(t, low_quantile, high_quantile, alpha=0.15)
        # plt.legend()

        # if 'weak' in alg_name:
        #     plt.figure(6)
        #     plt.title('Difference estimated reward random context {}'.format(other_context))
        #     for a in range(model.n_actions):
        #         plt.plot(t, np.mean(val.prod_scalar[:, :, a, other_context], axis=0), label=alg_name + ' arm {}'.format(a))
        #         low_quantile = np.quantile(val.prod_scalar[:, :, a, other_context], 0.1, axis=0)
        #         high_quantile = np.quantile(val.prod_scalar[:, :, a, other_context], 0.9, axis=0)
        #         plt.fill_between(t, low_quantile, high_quantile, alpha=0.15)
        #     plt.legend()
        #
        # if not 'weak' in alg_name:
        #     plt.figure(7)
        #     plt.title('Error biased env and learned env')
        #     for a in range(model.n_actions):
        #         error = np.linalg.norm(val.thetas[:, :, a] - model.thetas[target_arm]*(1 - attack_parameter), axis=2)
        #         plt.plot(t, np.mean(error, axis=0), label=alg_name + ' arm {}'.format(a))
        #         low_quantile = np.quantile(error, 0.1, axis=0)
        #         high_quantile = np.quantile(error, 0.9, axis=0)
        #         plt.fill_between(t, low_quantile, high_quantile, alpha=0.15)
        #     plt.legend()

        # plt.figure(8)
        # plt.title('Number of pulls target arm attack context')
        # plt.plot(t, np.mean(np.cumsum(val.target_draws == target_arm, axis=1), axis=0), label=alg_name + ' arm {}'.format(a))
        # low_quantile = np.quantile(np.cumsum(val.target_draws == target_arm, axis=1), 0.1, axis=0)
        # high_quantile = np.quantile(np.cumsum(val.target_draws == target_arm, axis=1), 0.9, axis=0)
        # plt.fill_between(t, low_quantile, high_quantile, alpha=0.15)
        # plt.legend()


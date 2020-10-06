# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import datetime
import json
import logging
import os
import pickle
from collections import namedtuple

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import sys
from math import sqrt


sys.path.append('/private/home/broz/workspaces/bandits_attacks')
from isoexp.contextual.contextual_linucb import ContextualLinearTS
from isoexp.contextual.contextual_models import DatasetModel, RandomContextualLinearArms

from isoexp.mab import contextual_arms
from isoexp.mab.contextual_mab_algs import contextEpsGREEDY, ContextualLinearBandit
import quadprog
from scipy.linalg import sqrtm
from scipy.optimize import minimize, linprog
import cvxpy as cp
from scipy import stats
from math import log

"""
TEST Linear Bandit 
"""

logging_period = 1000
def compute_regret(theta, context, a_t):
    D = np.dot(theta, context)
    return np.max(D) - D[a_t]


def work(nb_arms, noise, n_features, T, random_state, attack_frequency, alg_name, weak_attack=False,
         adversarial_xi=0.00001, method=None, sparse_attacks=False, simulator=None, target_arm=None, x_star=None, delta=0.99, reg_factor=0.1):
    # create model
    print(
        f"adversarial {attack_frequency}, xi {adversarial_xi}, weak_attack {weak_attack} method {method}")
    local_random = np.random.RandomState(random_state)
    if simulator is None:
        raise ValueError('No simulator')
    #     # real_theta = np.random.randn(nb_arms, n_features)
    #     # real_theta = np.random.uniform(low=1 / 2, high=3) * real_theta / np.linalg.norm(real_theta)
    #
    #     # simulator = contextual_arms.ContextualLinearMABModel(theta=real_theta, noise=noise, random_state=local_random)
    #     simulator = RandomContextualLinearArms(n_actions=nb_arms, n_features=n_features, noise=noise, bound_context=1)
    #     target_context = np.random.randint(low=0, high=len(simulator.context_lists))
    #     x_star = simulator.context_lists[target_context]
    #     means_x_star = np.dot(simulator.thetas, x_star)
    #
    #     target_arm = np.argmin(means_x_star)
    # # print('the simulator is {}'.format(simulator))

    simulator.local_random = local_random
    all_rewards = np.dot(simulator.context_lists, simulator.thetas.T)

    regret = []
    rewards = []
    norms = []
    attacks = []
    relative_attack_norm = []
    contexts_norms = []
    successful_attack = []
    failed_attack = []
    iteration = []
    cumulative_regret = []
    ratio_successful_attacks = []
    sum_attacks_norms = []
    nb_attacks_list = []
    inv_design_worst_ratio = []
    alg_names = []
    biases = []
    target_arm_chosen_list = []
    target_arm_chosen_count = 0
    x_star_appeared =0
    a_star_in_x_star=0
    a_star_in_x_star_list=[]
    x_star_appeared_list =[]
    TS_attacker=None
    if alg_name == 'eps_greedy':
            alg = contextEpsGREEDY(number_arms=simulator.n_actions, dimension=simulator.n_features, decrease_epsilon=True)
    elif alg_name == 'LinUCB':
            alg = ContextualLinearBandit(nb_arms=simulator.n_actions, dimension=simulator.n_features,
                                         reg_factor=reg_factor, delta=delta,
                                         bound_features=np.max(np.linalg.norm(simulator.thetas, axis=1)),
                                         noise_variance=noise, bound_context=simulator.bound_context)
    elif alg_name == 'LinTS':
            alg = ContextualLinearTS(nb_arms=simulator.n_actions, dimension=simulator.n_features,
                                     reg_factor=reg_factor,
                                     delta=delta,
                                     noise_variance=noise/5)
            TS_attacker = TS_relaxed_attacks_calculator(simulator, alg, T)
    else:
        raise ValueError(f'Unknown alg_name {alg_name}')
    cumulative_regret_t = 0
    n_successful_attacks = 0
    n_failed_attacks = 0
    attack_sums = 0
    nb_attacks = 0

    for t in tqdm(range(T)):
        context = simulator.get_context()
        context_norm = norm(context)
        if attack_frequency == 'target_arm':
            is_attacked = is_equal(context, x_star)
        else:
            attack_proba = 1 / sqrt(t + 1) if attack_frequency == 'decrease_sqrt' else attack_frequency
            is_attacked = local_random.rand() < attack_proba
        if is_attacked:
            predicted_best_arm = alg.get_action(context, deterministic=True)
            if sparse_attacks:
                # true_best_arm = np.argmax(simulator.theta.dot(context))
                if predicted_best_arm == target_arm:
                    # print("no need to attack")
                    n_successful_attacks += 1
                    attack = 0
                    attack_norm = 0
                else:
                    n_failed_attacks += 1
                    attack = compute_long_term_attack(simulator, predicted_best_arm, context, target_arm, all_rewards, factor=sparse_attacks)
                    attack_norm = norm(attack)
            else:
                estimated_rewards = alg.thetas.dot(context)
                if weak_attack:
                    attack, attack_norm, attack_succeeded = compute_weak_attack(adversarial_xi, alg, predicted_best_arm,
                                                                                context,
                                                                                estimated_rewards, nb_arms)
                else:
                    attack, attack_norm, attack_succeeded = compute_strong_attack(adversarial_xi, alg,
                                                                                  context,
                                                                                  estimated_rewards, method,
                                                                                  simulator.n_features,
                                                                                  simulator=simulator, target_arm=target_arm, x_star=x_star, attacker=TS_attacker)

            if attack_norm == float('inf'):
                attack = 0
                attack_norm = 0
        else:
            attack_norm = 0
            attack = 0

        if attack_norm > 0:
            nb_attacks += 1
        if attack_norm < float('inf'):
            attack_sums += attack_norm
        attacked_context = context + attack
        a_t = alg.get_action(attacked_context)
        if is_attacked and not sparse_attacks:
            if attack_succeeded:
                assert t <= nb_arms or 0 < attack_norm < float(
                    'inf'), 'The attack is seen as successful but is zero or of infinite norm, the attack was {}'.format(
                    attack)
                n_successful_attacks += 1
            else:
                n_failed_attacks += 1
        r_t = simulator.reward(context, a_t)
        regret_t = compute_regret(simulator.theta, context, a_t)
        alg.update(attacked_context, a_t, r_t)

        if is_equal(context, x_star):
            x_star_appeared += 1
            if a_t == target_arm:
                a_star_in_x_star+=1
        cumulative_regret_t += regret_t
        if a_t == target_arm:
            target_arm_chosen_count +=1
        if t % logging_period == 0:
            bias = (r_t - alg.thetas[a_t].dot(attacked_context)) / r_t
            norm_error = np.linalg.norm(alg.thetas - simulator.theta, 2)
            # logging
            worst_ratio = None
            for inv_a in alg.inv_design_matrices:
                for i, col in enumerate(inv_a):
                    ratio = abs(max(col) / col[i])
                    if worst_ratio is None or worst_ratio < ratio:
                        worst_ratio = ratio
            inv_design_worst_ratio.append(worst_ratio)
            regret.append(
                regret_t)  # simulator.best_expected_reward(context) - simulator.expected_reward(action=a_t, context=context)
            norms.append(norm_error)
            rewards.append(r_t)
            attacks.append(attack_norm)
            iteration.append(t)
            relative_attack_norm.append(norm(attacked_context) / context_norm)
            contexts_norms.append(context_norm)
            cumulative_regret.append(cumulative_regret_t)
            ratio_successful_attacks.append(n_successful_attacks / (
                    n_failed_attacks + n_successful_attacks) if n_failed_attacks + n_successful_attacks else 0)
            successful_attack.append(n_successful_attacks)
            failed_attack.append(n_failed_attacks)
            sum_attacks_norms.append(attack_sums)
            nb_attacks_list.append(nb_attacks)
            alg_names.append(alg_name)
            biases.append(bias)
            x_star_appeared_list.append(x_star_appeared)
            target_arm_chosen_list.append(target_arm_chosen_count)
            a_star_in_x_star_list.append(a_star_in_x_star)

        logging.info(f"Iteration {t}, regret {regret_t}, reward{r_t}, norm error {norm_error}")

    return {'iteration': iteration, "regret": regret, 'cumulative_regret': cumulative_regret, "rewards": rewards,
            "norm_errors": norms, "attacks": attacks, 'target_arm_chosen': target_arm_chosen_list,
            "relative_attack_norm": relative_attack_norm, 'contexts_norms': contexts_norms,
            'successful_attack': ratio_successful_attacks, 'xi': adversarial_xi, 'biases': biases,
            'attack_frequency': attack_frequency, 'sum_attacks_norms': sum_attacks_norms, 'weak_attack': weak_attack,
            'x_star_appearances':x_star_appeared_list, 'a_star_in_x_star': a_star_in_x_star_list,
            'method': method, 'sparse_attacks': sparse_attacks, "nb_attacks": nb_attacks_list,
            'n_successful_attack': successful_attack, 'n_failed_attack': failed_attack,
            'design_mat_worse_ratio': inv_design_worst_ratio, 'alg_names': alg_names}, simulator


def is_equal(context, x_star):
    if x_star is None:
        return False
    return norm(context - x_star) < 1e-8


def compute_short_attack_linUCB(dimension, alg, a_star, x_star, slack=10 ** -5, relaxed=False):
    func = lambda delta: np.linalg.norm(delta)/2
    theta_a_star = alg.thetas[a_star]
    P_a_star = sqrtm(alg.inv_design_matrices[a_star])
    betas = alg.alpha()
    constraints_list = []
    for a in range(len(alg.thetas)):
        if a != a_star:
            theta = alg.thetas[a]
            P = sqrtm(alg.inv_design_matrices[a])
            if not(relaxed):
                temp_constraint = lambda delta, P=P, P_a_star=P_a_star, theta=theta, theta_a_star=theta_a_star, beta=betas[a], beta_a_star=betas[a_star]: \
                    -((theta - theta_a_star).dot((x_star + delta)) + beta * np.linalg.norm(P.dot((x_star + delta)))
                    - beta_a_star * np.linalg.norm(P_a_star.dot((x_star + delta))) + slack)
            else:
                temp_constraint = lambda delta, P=P, P_a_star=P_a_star, theta=theta, theta_a_star=theta_a_star, beta=betas[a], beta_a_star=betas[a_star]: \
                    -((theta - theta_a_star).dot((x_star + delta)) + beta * np.linalg.norm(P.dot((x_star + delta))) + slack)
            temp_cons = {'type': 'ineq', 'fun': temp_constraint}
            constraints_list.append(temp_cons)
    cons = tuple(constraints_list)
    res = minimize(func, -x_star, method='SLSQP', constraints=cons)
    # print(res.message)
    try:
        epsilon = res.x
    except KeyboardInterrupt:
        raise
    except:
        epsilon = np.zeros((dimension,))
    # print('Epsilon =', epsilon)
    # for a in range(len(constraints_list)):
    #     theta = alg.thetas[a]
    #     P = sqrtm(alg.inv_design_matrices[a])
    #     print('Constraints for arm {}'.format(a), constraints_list[a]['fun'](x_star + epsilon))
    if epsilon is None:
        return np.zeros((dimension,)), 0, False
    return epsilon, norm(epsilon), norm(epsilon) > 0

def compute_relaxed_attack(dimension, alg, a_star, x_star, slack=10 ** -5):
    delta = cp.Variable(dimension)
    obj = cp.Minimize(cp.quad_form(delta, np.eye(dimension))/2)
    theta_a_star = alg.thetas[a_star]
    betas = alg.alpha()
    constraints = []
    for a in range(len(alg.thetas)):
        if a != a_star:
            theta = alg.thetas[a]
            P = sqrtm(alg.inv_design_matrices[a])
            temp_constraint = (theta - theta_a_star) @ (x_star + delta) + betas[a] * cp.norm2(P @ (x_star + delta))
            constraints.append(temp_constraint + slack <= 0)
    prob = cp.Problem(obj, constraints)
    try:
        out = prob.solve(solver='SCS', max_iters=10000,)
        epsilon = delta.value
    except KeyboardInterrupt:
        raise
    except:
        print('Exception')
        epsilon = np.zeros((dimension,))
    if epsilon is None:
        return np.zeros((dimension,)), 0, False
    # if norm(epsilon > 0):
    #     margin = (theta - theta_a_star) @ (x_star + epsilon) + betas[a] * np.linalg.norm(np.dot(np.array(sqrtm(alg.inv_design_matrices[a])) ,(x_star + epsilon))) #np.sqrt(np.dot(x_star + epsilon, alg.inv_design_matrices[a] @ (x_star + epsilon)))
    #     # print(f'the margin is {margin}')
    #     if margin > 0 :
    #         print('the margin was negative, {}, norm eps {}, norm x {}'.format(out, norm(epsilon), norm(x_star)))
    return epsilon, norm(epsilon), norm(epsilon) > 0


class TS_relaxed_attacks_calculator:
    def __init__(self, simulator, alg, T):
        delta_zero = 0.95
        sigma = alg.sigma
        nu = sigma * 3 * sqrt(simulator.n_features * log(T/delta_zero))
        self.cste = nu * stats.norm.ppf(1 - delta_zero / (simulator.thetas.shape[0] -1))

    def compute_relaxed_attack(self, dimension, alg, a_star, x_star, slack=10 ** -5):
        delta = cp.Variable(dimension)
        obj = cp.Minimize(cp.quad_form(delta, np.eye(dimension))/2)
        theta_a_star = alg.thetas[a_star]
        constraints = []
        for a in range(len(alg.thetas)):
            if a != a_star:
                theta = alg.thetas[a]
                P = sqrtm(alg.inv_design_matrices[a] + alg.inv_design_matrices[a_star])
                temp_constraint = (theta - theta_a_star) @ (x_star + delta) + self.cste * cp.norm(P @ (x_star + delta))
                constraints.append(temp_constraint + slack <= 0)
        prob = cp.Problem(obj, constraints)
        try:
            prob.solve()#(feastol=1e-11, feastol_inacc=1e-11)
            epsilon = delta.value
        except KeyboardInterrupt:
            raise
        except:
            print('Exception')
            epsilon = np.zeros((dimension,))
        # print('epsilon =', epsilon)
        if epsilon is None:
            return np.zeros((dimension,)), 0, False
        return epsilon, norm(epsilon), norm(epsilon) > 0


def compute_strong_attack(adversarial_xi, alg, context, estimated_rewards, method, n_features, simulator, target_arm, x_star, attacker=None):
    # worst_arm = np.argmin(estimated_rewards)
    if method == 'linUCB_Relaxed':
        alg.get_action(context)
        attack, attack_norm, attack_succeeded = compute_relaxed_attack(simulator.n_features, alg, target_arm, context, slack=10 ** -9)
        # attack, attack_norm, attack_succeeded  = compute_short_attack_linUCB(simulator.n_features, alg, target_arm, x_star, slack=10 ** -10 , relaxed=False)
        # if attack_succeeded:
        #     print(f'attack succeeded {attack_norm}')
        #     new_chosen = alg.get_action(context + attack, deterministic=True)

            # if new_chosen != target_arm:
            #     new_context = context + attack
            #     print(f'algorithm chose arm {new_chosen} instead of {target_arm}')
            #     print(
            #         f'the scores were {alg.thetas[target_arm].dot(new_context) + alg.alpha()[target_arm] * np.sqrt(np.dot(new_context, np.dot(alg.inv_design_matrices[target_arm], new_context)))} vs {alg.thetas[new_chosen].dot(new_context) + alg.alpha()[new_chosen] * np.sqrt(np.dot(new_context, np.dot(alg.inv_design_matrices[new_chosen], new_context)))}, {norm(context+attack)}')
            #     print(
            #         f'with just attack the scores were {alg.thetas[target_arm].dot(attack)} vs {alg.thetas[new_chosen].dot(attack)}')
            #     # raise ValueError('Wrong chosen arm')
        return attack, attack_norm, attack_succeeded
    elif method == 'linUCB_full':
        return compute_short_attack_linUCB(simulator.n_features, alg, target_arm, context, slack=10 ** -3, relaxed=False)
    elif method == 'TS_Relaxed':
        assert(attacker is not None), "Should pass an attacker to attack LinTS"
        return attacker.compute_relaxed_attack(simulator.n_features, alg, target_arm, context, slack=10 ** -10)

    elif method == 'quadprog':
        try:
            attack = quadprog_solve_qp(n_features, thetas=alg.thetas, arm_to_select=target_arm, context=context,
                                       adversarial_xi=10**-5)
            attack_norm = norm(attack)
            attack_succeeded = True

        except ValueError:
            attack = 0
            attack_norm = float('inf')
            attack_succeeded = False
            return attack, attack_norm, attack_succeeded
    # elif method == 'heuristic':
    #     best_arm = np.argmax(estimated_rewards)
    #     attack, attack_norm = compute_attack_for_arm(worst_arm, best_arm, alg.thetas, estimated_rewards,
    #                                                  adversarial_xi)
    #     attack_succeeded = alg.estimated_best_arm(context + attack) == worst_arm
    else:
        assert False, f'Unkown method for targeted attacks: {method}'
    return attack, attack_norm, attack_succeeded

def compute_long_term_attack(simulator, action, context, a_star, all_rewards, slack=10 ** -3, factor=2):
    if action != a_star:  # and np.linalg.norm(context - x_star) < 10**-5:
        # worst_ratio1 = (all_rewards[:, action] / all_rewards[:, a_star]).max()
        worst_ratio = 1 / all_rewards[:, a_star].min()
        # print(worst_ratio1, worst_ratio)
        # assert(worst_ratio1 <= worst_ratio), 'there is a reward that is greater than 1'
        delta = factor * worst_ratio
        # print(f'delta: {delta}')
        delta = max(delta, 1)
        # delta = np.maximum(2*np.dot(model.thetas[action], x_star)/np.dot(model.thetas[a_star], x_star), 1)
        epsilon = (delta - 1) * context
        return epsilon
    else:
        return np.zeros((simulator.n_features,))


def compute_weak_attack(adversarial_xi, alg, best_arm, context, estimated_rewards, nb_arms):
    possible_attacks = [compute_attack_for_arm(arm, best_arm, alg.thetas, estimated_rewards, adversarial_xi) for arm
                        in range(nb_arms) if arm != best_arm]
    attack, attack_norm = possible_attacks[np.argmin([att[1] for att in possible_attacks])]
    attack_succeeded = alg.estimated_best_arm(context + attack) != alg.estimated_best_arm(context)
    return attack, attack_norm, attack_succeeded


def norm(vector):
    return np.linalg.norm(vector, 2)


def compute_attack_for_arm(chosen_arm, best_arm, thetas, estimated_rewards, adversarial_xi):
    attack_direction = thetas[chosen_arm] - thetas[best_arm]
    norm = attack_direction.dot(attack_direction)
    if norm == 0:
        return 0, float('inf')
    attack_norm = (estimated_rewards[best_arm] - estimated_rewards[chosen_arm] + adversarial_xi) / norm
    attack = attack_norm * attack_direction
    return attack, attack_norm


def generate_context(n_features, low=-3, high=3):
    context = np.random.randn(n_features)
    context = np.random.uniform(low=low, high=high) * context / np.linalg.norm(context)
    return context


def quadprog_solve_qp(n_features, thetas, arm_to_select, context, adversarial_xi):
    qp_G = np.identity(n_features)  # make sure P is symmetric
    qp_a = np.zeros_like(context)
    # no equality constraint
    constraints_lines = np.delete(thetas - thetas[arm_to_select], arm_to_select, axis=0)
    qp_C = - constraints_lines.T
    qp_b = constraints_lines.dot(context) + adversarial_xi
    meq = 0
    return quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]


# if __name__ == '__main__':
#
#     PARALLEL = True
#     print("PARALLEL: {}".format(PARALLEL))
#
#     MABResults = namedtuple('MABResults', 'regret, cum_rewards, norm_errors')
#
#     random_state = np.random.randint(0, 123123)
#     np.random.seed(random_state)
#     local_random = np.random.RandomState(random_state)
#
#     print("seed: {}".format(random_state))
#
#     K = 10
#     n_features = 30
#     a_noise = 0.1
#
#     T = 5 * 10 ** 6  # horizon
#     nb_simu = 15  # 48 * 5 #240
#     adversarial_xi = 0.0001
#
#     attack_frequencies = [1.0, 0.0]  # [1.0, 'decrease_sqrt', 0]
#     algo_names = ['LinUCB', 'eps-greedy', 'LinTS']
#     weak_attacks_list = [False]  # [False, True] #
#     methods_to_test = [None]  # ['quadprog', 'heuristic']
#     sparse_factors = [2.0]
#     results = []
#
#     sparse_attacks = None
#     movielens = True
#     jester = False
#     dataset_model = movielens or jester
#     assert(not(movielens and jester)), "cannot use both movielens and jester"
#     if dataset_model:
#         if movielens:
#             simulator = DatasetModel(os.path.abspath('examples/movielens/Vt_movielens.csv'), user_csvfile=os.path.abspath("examples/movielens/U.csv"), arms_limit=100)
#         elif jester:
#             simulator = DatasetModel(os.path.abspath("examples/jester/Vt_jester.csv"), user_csvfile=os.path.abspath('examples/jester/U.csv'))
#         else:
#             print('Issue, should use a dataset that isn\'t jester or movielens')
#             exit(0)
#         # target_context = np.random.randint(low=0, high=len(simulator.context_lists))
#         # x_star = simulator.context_lists[target_context]
#         means_x_star = np.dot(simulator.context_lists, simulator.thetas.T).mean(axis=0)
#         target_arm = np.argmin(means_x_star)
#     else:
#         simulator = None
#         target_arm = None
#
#     settings = {
#         "T": T,
#         "nb_simu": nb_simu,
#         "random_state": random_state,
#         "K": simulator.n_actions if simulator else K,
#         "dimension": simulator.n_features if simulator else n_features,
#         'attack_frequencies': attack_frequencies,
#         'weak_attacks': weak_attacks_list,
#         'methods_to_test': methods_to_test,
#         'adversarial_xi': adversarial_xi,
#         'sparse_factors': sparse_factors,
#         'target_arm': target_arm,
#     }
#     weak_attack=False
#     method=None
#     if PARALLEL:
#         import multiprocessing
#         work_to_be_done = []
#         # for attack_frequency in attack_frequencies:
#         #     sparse_factors_to_test = sparse_factors if attack_frequency != 0 else [False]
#         #     for sparse_attacks in sparse_factors_to_test:
#         #         method = None
#         #         weak_attack = False
#         #         for weak_attack in weak_attacks_list if attack_frequency else [True]:
#         #             methods_to_test_list = methods_to_test if not weak_attack and attack_frequency != 0 else [
#         #                 'quadprog']
#         #             for method in methods_to_test_list:
#         #                 for xi in adversarial_xi:
#
#         for alg_name in algo_names:
#             for sim_index in range(nb_simu):
#                                 work_to_be_done.append(
#                                     (attack_frequency, sparse_attacks, weak_attack, method, adversarial_xi, sim_index, alg_name))
#
#         for sim_index in range(nb_simu):
#         #     work_to_be_done.append((0.2, 10, False, 'quadprog', xi, sim_index))
#             work_to_be_done.append((0.2, 10, False, 'quadprog', adversarial_xi[0], sim_index))
#         settings['work_list'] = work_to_be_done
#         num_cores = multiprocessing.cpu_count()
#         results.append(Parallel(n_jobs=num_cores, verbose=1)(
#             delayed(work)(nb_arms=K, noise=a_noise, n_features=n_features, T=T, random_state=random_state + sim_index,
#                            attack_frequency=attack_frequency,alg_name=alg_name,
#                           weak_attack=weak_attack, adversarial_xi=xi, method=method,
#                           sparse_attacks=sparse_attacks, simulator=simulator, target_arm=target_arm) for
#             attack_frequency, sparse_attacks, weak_attack, method, xi, sim_index, alg_name in work_to_be_done))
#     else:
#         #     for decrease_epsilon in [True, False]:
#         for attack_frequency in [0]:  # [1.0,0.1, 0]:
#             weak_attack = False
#             for k in tqdm(range(nb_simu)):
#                 ret = work(nb_arms=K, noise=a_noise, n_features=n_features, T=T, random_state=random_state + k,
#                            attack_frequency=attack_frequency,
#                            weak_attack=weak_attack)
#                 results.append(ret)
#
#
#     id = '{}_{:%Y%m%d_%H%M%S}_{}'.format('jester' if jester else 'movilens' if movielens else 'simulation', datetime.datetime.now(), '_Contextual_linear')
#     pickle_name = "{}_{}_linear_results.pickle".format(id, "PAR" if PARALLEL else "SEQ")
#     print(pickle_name)
#     with open(pickle_name, "wb") as f:
#         pickle.dump(results, f)
#     # with open("{}_{}_linear_settings.json".format(id, "PAR" if PARALLEL else "SEQ"), "w+") as f:
#     #     json.dump(settings, f)

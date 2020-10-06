# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from isoexp.contextual.contextual_models import RandomContextualLinearArms
from isoexp.contextual.contextual_linucb import *
from tqdm import trange
from collections import namedtuple
from cycler import cycler
import matplotlib.pyplot as plt
import cvxpy as cp

class exp(object):
    def __init__(self, nb_arms, type='random', a_star = 0):
        self.K = nb_arms
        self.type= type
        self.a_star = a_star

    def get_action(self, context):
        if self.type == 'random':
            return np.ones((self.K,))/self.K
        elif self.type == 'optimal':
            means = np.dot(model.thetas, context)
            a = np.argmax(means)
            proba = np.zeros((self.K,))
            proba[a] = 1
            return proba
        else:
            proba = np.zeros((self.K,))
            proba[self.a_star] = 1
            return proba


n = 6  # Number of colors
new_colors = [plt.get_cmap('Set1')(1. * i / n) for i in range(n)]
linestyle_cycler = cycler('linestyle', ['-', '--', ':', '-.', '-', '--'])
plt.rc('axes', prop_cycle=(cycler('color', new_colors) + linestyle_cycler))
plt.rc('lines', linewidth=2)

MABResults = namedtuple('MABResults', 'regret, attack_cond, target_draws, thetas, prod_scalar')
seed = np.random.randint(0, 10 ** 5)
print('seed = ', seed)
noise = 0.1
attack_parameter = 1/2
model = RandomContextualLinearArms(n_actions=5, n_features=10, noise=noise, random_state=seed, bound_context=1)
theta_bound = np.max(np.linalg.norm(model.thetas, 2, axis=(1)))
# target_context = np.random.randint(low=0, high=len(model.context_lists))
# other_context = np.random.randint(low=0, high=len(model.context_lists))
# while other_context == target_context:
#     other_context = np.random.randint(low=0, high=len(model.context_lists))
# target_arm = np.random.randint(low=0, high=model.n_actions)
target_arm = np.argmax(np.dot(model.thetas, model.context_lists[-1]))
T = 5000
nb_simu = 30
M = 10
print('a_star=', target_arm)
eta = np.sqrt(2*np.log(M)/(T*model.n_actions))
experts = []
for i in range(M-2):
    experts.append(exp(nb_arms=model.n_actions, type='random'))
experts.append(exp(nb_arms=model.n_actions, type='optimal'))
experts.append(exp(nb_arms=model.n_actions, type='', a_star=int(target_arm)))

algorithms = {
    'Exp4': Exp4(nb_arms=model.n_actions, dimension=model.n_features, experts=experts, eta=eta, gamma=10**-3)
}
results = []
for alg_name, alg in algorithms.items():
    regret = np.zeros((nb_simu, T))
    draws = np.zeros((nb_simu, T))
    epsilon_norm = np.zeros((nb_simu, T))
    thetas_alg = np.zeros((nb_simu, T, model.n_actions, model.n_features))
    prod_scalar = np.zeros((nb_simu, T, model.n_actions, model.n))
    for k in trange(nb_simu, desc='Simulating {}'.format(alg_name)):
        alg.reset()
        for t in range(T):
            context = model.get_context()
            a_t = alg.get_action(context)
            r_t = model.reward(context, a_t)
            attack_t = 0
            epsilon_norm[k, t] = np.abs(attack_t)
            alg.update(context, a_t, r_t + attack_t)
            # try:
            #     thetas_alg[k, t] = alg.thetas_hat
            # except:
            #     pass
            # for a in range(model.n_actions):
            #     for i, x in enumerate(model.context_lists):
            #         p = np.dot(alg.thetas_hat[a], x) - (1 - attack_parameter)*np.dot(model.thetas[target_arm], x)
            #         prod_scalar[k, t, a, i] = p
            regret[k, t] = model.best_arm_reward(context) - np.dot(model.thetas[a_t], context)
            draws[k, t] = a_t

    results += [(alg_name, MABResults(regret=regret, attack_cond=epsilon_norm, target_draws=draws,
                                      thetas=thetas_alg, prod_scalar=prod_scalar))]

print('Target arm =', target_arm)
print('draws = ', np.mean(np.cumsum(draws == target_arm, axis=1),axis=0))

for i,(alg_name, val) in enumerate(results):
    mean_regret = np.mean(val.regret.cumsum(axis=1), axis=0)
    t = np.linspace(0, T, T, dtype='int')
    low_quantile = np.quantile(val.regret.cumsum(axis=1), 0.1, axis=0)
    high_quantile = np.quantile(val.regret.cumsum(axis=1), 0.9, axis=0)

    plt.figure(0)
    plt.title('Cumulative Regret')
    plt.plot(mean_regret, label=alg_name)
    plt.fill_between(t, low_quantile, high_quantile, alpha=0.15)
    plt.legend()

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
    plt.figure(4)
    plt.title('Error true env and learned env')
    for a in range(model.n_actions):
        error = np.linalg.norm(val.thetas[:, :, a] - model.thetas[a], axis=2)
        plt.plot(t, np.mean(error, axis=0), label=alg_name + ' arm {}'.format(a))
        low_quantile = np.quantile(error, 0.1, axis=0)
        high_quantile = np.quantile(error, 0.9, axis=0)
        plt.fill_between(t, low_quantile, high_quantile, alpha=0.15)
    plt.legend()

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


plt.show()

#print('Target arms=', np.mean(np.cumsum(nb_target_arms,axis=1)))
#print('Attack needed arms=', np.mean(np.cumsum(nb_attack_needed,axis=1)))
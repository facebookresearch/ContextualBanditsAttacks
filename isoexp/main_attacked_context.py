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

def compute_attack(model, action, context, a_star, x_star, slack=10**-3):
    if action != a_star:# and np.linalg.norm(context - x_star) < 10**-5:
        delta = np.maximum(2*np.dot(model.thetas[action], x_star)/np.dot(model.thetas[a_star], x_star), 1)
        epsilon = (delta - 1)*context
        return epsilon
    else:
        return np.zeros((model.n_features,))

n = 6  # Number of colors
new_colors = [plt.get_cmap('Set1')(1. * i / n) for i in range(n)]
linestyle_cycler = cycler('linestyle', ['-', '--', ':', '-.', '-', '--'])
plt.rc('axes', prop_cycle=(cycler('color', new_colors) + linestyle_cycler))
plt.rc('lines', linewidth=2)

MABResults = namedtuple('MABResults', 'regret, attack_cond, target_draws, thetas, prod_scalar, context_norm')
seed = np.random.randint(0, 10 ** 5)
print('seed = ', seed)
noise = 0.1
model = RandomContextualLinearArms(n_actions=3, n_features=10, noise=noise, random_state=seed, bound_context=1)
model.thetas = model.thetas
theta_bound = np.max(np.linalg.norm(model.thetas, 2, axis=(1)))
target_context = np.random.randint(low=0, high=len(model.context_lists))
other_context = np.random.randint(low=0, high=len(model.context_lists))
while other_context == target_context:
    other_context = np.random.randint(low=0, high=len(model.context_lists))
x_star = model.context_lists[target_context]
means_x_star = np.dot(model.thetas, x_star)
#target_arm = np.random.randint(low=0, high=model.n_actions)
target_arm = np.argmin(means_x_star)
T = 1*10**4
nb_simu = 5
print('a star=', target_arm)
print('x_star', x_star)
print('means for context x_star:', np.dot(model.thetas, x_star))

algorithms = {
    'LinUCB': ContextualLinearBandit(nb_arms=model.n_actions, dimension=model.n_features,
                                     reg_factor=0.1, delta=0.99,
                                     bound_features=np.max(np.linalg.norm(model.thetas, axis=1)),
                                     noise_variance=noise, bound_context=model.bound_context),
    'LinUCB-Attacked': ContextualLinearBandit(nb_arms=model.n_actions, dimension=model.n_features,
                                     reg_factor=0.1, delta=0.99,
                                     bound_features=np.max(np.linalg.norm(model.thetas, axis=1)),
                                     noise_variance=noise, bound_context=model.bound_context)
}
results = []
for alg_name, alg in algorithms.items():
    regret = np.zeros((nb_simu, T))
    nb_target_arms = np.zeros((nb_simu, T))
    nb_attack_needed = np.zeros((nb_simu, T))
    attack_condition = np.zeros((nb_simu, T))
    draws = np.zeros((nb_simu, T))
    context_norm = draws.copy()
    epsilon_norm = np.zeros((nb_simu, T))
    thetas_alg = np.zeros((nb_simu, T, model.n_actions, model.n_features))
    prod_scalar = np.zeros((nb_simu, T, model.n_actions, model.n))
    for k in trange(nb_simu, desc='Simulating {}'.format(alg_name)):
        alg.reset()
        for t in range(T):
            context = model.get_context()
            a_t = alg.get_action(context)
            old_context = context
            if alg_name == 'LinUCB-Attacked':
                epsilon = compute_attack(model, a_t, context, target_arm, x_star)
                context = context + epsilon
                alg.iteration -= 1
                a_t = alg.get_action(context)
                epsilon_norm[k, t] = np.linalg.norm(epsilon)
            thetas_alg[k, t] = alg.thetas_hat
            for a in range(model.n_actions):
                for i, x in enumerate(model.context_lists):
                    p = np.dot(model.thetas[a], x) - np.dot(alg.thetas_hat[a], x)
                    prod_scalar[k, t, a, i] = p
            r_t = model.reward(old_context, a_t)
            alg.update(context, a_t, r_t)
            regret[k, t] = model.best_arm_reward(old_context) - np.dot(model.thetas[a_t], old_context)
            context_norm[k, t] = np.linalg.norm(context)
            draws[k, t] = a_t
    results += [(alg_name, MABResults(regret=regret, attack_cond=attack_condition, target_draws=draws,
                                      thetas=thetas_alg, prod_scalar=prod_scalar, context_norm=context_norm))]

print('Target arm =', target_arm)

print('draws = ', np.mean(np.cumsum(draws == target_arm, axis=1),axis=0))
for i, (alg_name, val) in enumerate(results):
    mean_regret = np.mean(val.regret.cumsum(axis=1), axis=0)
    t = np.linspace(0, T, T, dtype='int')

    low_quantile = np.quantile(val.regret.cumsum(axis=1), 0.1, axis=0)
    high_quantile = np.quantile(val.regret.cumsum(axis=1), 0.9, axis=0)

    plt.figure(0)
    plt.title('Regret Attacked context')
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

    if 'Attacked' in alg_name:
        plt.figure(2)
        plt.title('Cumulative attack norm attacked context')
        plt.plot(np.mean(np.cumsum(epsilon_norm, axis=1), axis=0), label=alg_name)
        low_quantile = np.quantile(np.cumsum(epsilon_norm, axis=1), 0.1, axis=0)
        high_quantile = np.quantile(np.cumsum(epsilon_norm, axis=1), 0.9, axis=0)
        plt.fill_between(t, low_quantile, high_quantile, alpha=0.15)
        plt.legend()


    plt.figure(4)
    plt.title('Error true env and learned env attack context')
    for a in range(model.n_actions):
        error = np.linalg.norm(val.thetas[:, :, a] - model.thetas[a], axis=2)
        plt.plot(t, np.mean(error, axis=0), label=alg_name + ' arm {}'.format(a))
        low_quantile = np.quantile(error, 0.1, axis=0)
        high_quantile = np.quantile(error, 0.9, axis=0)
        plt.fill_between(t, low_quantile, high_quantile, alpha=0.15)
    plt.legend()


    plt.figure(7)
    plt.title('Error biased env and learned env attack context')
    for a in range(model.n_actions):
        error = np.linalg.norm(val.thetas[:, :, a] - model.thetas[a]/np.maximum(2*np.dot(model.thetas[a], x_star)/np.dot(model.thetas[target_arm], x_star), 1), axis=2)
        plt.plot(t, np.mean(error, axis=0), label=alg_name + ' arm {}'.format(a))
        low_quantile = np.quantile(error, 0.1, axis=0)
        high_quantile = np.quantile(error, 0.9, axis=0)
        plt.fill_between(t, low_quantile, high_quantile, alpha=0.15)
    plt.legend()

    plt.figure(5)
    plt.title('Difference estimated reward for target context {}'.format(target_context))
    for a in range(model.n_actions):
        plt.plot(t, np.mean(val.prod_scalar[:, :, a, target_context], axis=0), label=alg_name + ' arm {}'.format(a))
        low_quantile = np.quantile(val.prod_scalar[:, :, a, target_context], 0.1, axis=0)
        high_quantile = np.quantile(val.prod_scalar[:, :, a, target_context], 0.9, axis=0)
        plt.fill_between(t, low_quantile, high_quantile, alpha=0.15)
    plt.legend()


    plt.figure(6)
    plt.title('Difference estimated reward for a random non target context {}'.format(other_context))
    for a in range(model.n_actions):
        plt.plot(t, np.mean(val.prod_scalar[:, :, a, other_context], axis=0), label=alg_name + ' arm {}'.format(a))
        low_quantile = np.quantile(val.prod_scalar[:, :, a, other_context], 0.1, axis=0)
        high_quantile = np.quantile(val.prod_scalar[:, :, a, other_context], 0.9, axis=0)
        plt.fill_between(t, low_quantile, high_quantile, alpha=0.15)
    plt.legend()

    plt.figure(8)
    plt.title('Number of pulls target arm attack context')
    plt.plot(t, np.mean(np.cumsum(val.target_draws == target_arm, axis=1), axis=0), label=alg_name + ' arm {}'.format(a))
    low_quantile = np.quantile(np.cumsum(val.target_draws == target_arm, axis=1), 0.1, axis=0)
    high_quantile = np.quantile(np.cumsum(val.target_draws == target_arm, axis=1), 0.9, axis=0)
    plt.fill_between(t, low_quantile, high_quantile, alpha=0.15)
    plt.legend()

plt.show()

#print('Target arms=', np.mean(np.cumsum(nb_target_arms,axis=1)))
#print('Attack needed arms=', np.mean(np.cumsum(nb_attack_needed,axis=1)))
# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
from isoexp.linear.linearbandit import EfficientLinearBandit, LinearBandit, LinPHE
from isoexp.conservative.linearmabs import EfficientConservativeLinearBandit, SafetySetCLUCB
from isoexp.linear.linearmab_models import RandomLinearArms, DiffLinearArms, OtherArms, CircleBaseline, LinPHEModel
from matplotlib import rc
from joblib import Parallel, delayed
from isoexp.linear.coldstart import ColdStartFromDatasetModel
import os

rc('text', usetex=True)

import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import namedtuple

MABResults = namedtuple('MABResults', 'regret,norm_error, cum_rewards')

random_state = np.random.randint(0, 123123)


NOISE = 0.1
#model = RandomLinearArms(n_actions=300, n_features=100, noise=NOISE, bound_features = 5, bound_theta = 3)
model = ColdStartFromDatasetModel(csvfile=os.path.abspath('jester/Vt_jester.csv'), noise=NOISE)
theta_bound = np.linalg.norm(model.theta, 2)
means = np.dot(model.features, model.theta)
print(means)
idxs = np.argsort(means)
#baseline = np.random.randint(0, model.n_actions - 1)
baseline = idxs[-5]
mean_baseline = means[baseline]
optimal_arm = np.argmax(means)
PARALLEL = False

n_a = model.n_actions
d = model.n_features
T = 20000
batch_horizon = int(T*0.2)
nb_simu = 10
alpha = 0.1
algorithms = { 
        'EfficientLinearBandit': EfficientLinearBandit(arm_features=model.features,
                                         reg_factor=1.,
                                         delta=0.01,
                                         noise_variance=NOISE,
                                         bound_theta=theta_bound)
}
conservative_algorithms = {
#            'CLUCB-new': EfficientConservativeLinearBandit(model.features, baseline, mean_baseline,
#                 bound_theta=theta_bound, noise_variance=NOISE,
#                 reg_factor=1., delta=0.01, conservative_level=alpha, 
#                  version = 'new', oracle = False, means = means, 
#                  batched = False, check_every = batch_horizon, positive = True),
#            'CLUCB-old': EfficientConservativeLinearBandit(model.features, baseline, mean_baseline,
#                 bound_theta=theta_bound, noise_variance=NOISE,
#                 reg_factor=1., delta=0.01, conservative_level=alpha, 
#                  version = 'old', oracle = False, means = means, 
#                  batched = False, check_every = batch_horizon, positive = True),
#            'SafetySet-Old' : SafetySetCLUCB(model.features, baseline, mean_baseline,
#                 bound_theta=theta_bound, noise_variance=NOISE,
#                 reg_factor=1., delta=0.01, conservative_level=alpha, 
#                  version = 'old', batched = False, check_every = batch_horizon, positive = True, verbose = False),
#            'SafetySet-new' :  SafetySetCLUCB(model.features, baseline, mean_baseline,
#                 bound_theta=theta_bound, noise_variance=NOISE,
#                 reg_factor=1., delta=0.01, conservative_level=alpha, 
#                  version = 'new', oracle = False, batched = False, check_every = batch_horizon, means = means, 
#                  verbose = False, positive = True)
}

algorithms = {**algorithms, **conservative_algorithms}
if PARALLEL:
    import multiprocessing

    num_cores = multiprocessing.cpu_count()


    def work(alg_name, alg):

        regret = np.zeros((nb_simu, T))
        norms = np.zeros((nb_simu, T))
        cond = regret.copy()

        for k in tqdm(range(nb_simu)):

            alg.reset()

            for t in range(T):
                a_t = alg.get_action()
                # print(a_t)
                r_t = model.reward(a_t)
                alg.update(a_t, r_t)
                cond[k, t] = means[a_t] - (1-alpha)*mean_baseline
                regret[k, t] = model.best_arm_reward() - r_t
                if hasattr(alg, 'theta_hat'):
                    norms[k, t] = np.linalg.norm(alg.theta_hat - model.theta, 2)

        # results[alg_name] = \
        return alg_name, MABResults(regret=regret, norm_error=norms, cum_rewards = cond)


    results = Parallel(n_jobs=num_cores, verbose=1)(
        delayed(work)(alg_name, algorithms[alg_name]) for alg_name in algorithms.keys())

else:
    from tqdm import trange
    results = []
    for alg_name in algorithms.keys():

        

        regret = np.zeros((nb_simu, T))
        norms = np.zeros((nb_simu, T))
        cond = np.zeros((nb_simu, T))
        nb = 0
        draws = 0*regret

        for k in tqdm(range(nb_simu), desc='Simulating {}'.format(alg_name)):

            
            alg = algorithms[alg_name]
            alg.reset()

            for t in trange(T):
                a_t = alg.get_action()
                r_t = model.reward(a_t) 
                cond[k, t] = means[a_t] - (1-alpha)*mean_baseline
                alg.update(a_t, r_t)
                draws[k,t] = a_t

                if a_t == baseline:
                    nb += 1

                regret[k, t] = model.best_arm_reward() - r_t
                if hasattr(alg, 'theta_hat'):
                    norms[k, t] = np.linalg.norm(alg.theta_hat - model.theta, 2)
        results += [(alg_name, MABResults(regret=regret, norm_error=norms, cum_rewards=cond.cumsum(axis = 1)))]
#%%
plt.figure(1, figsize=(10, 10))
plt.figure(2, figsize=(10, 10))
for alg_name, val in results :
    temp = val.regret
    temp = temp.cumsum(axis = 1)
    mean_regret = np.mean(temp, axis=0)
    mean_norms = np.mean(val.norm_error, axis=0)
    low_quantile = np.quantile(temp, 0.000, axis=0)
    high_quantile = np.quantile(temp, 1, axis=0)
    condition_satisfied = np.mean(val.cum_rewards, axis=0)
    low_quantile_condition = np.quantile(val.cum_rewards, 0.25, axis=0)
    high_quantile_condition = np.quantile(val.cum_rewards, 0.75, axis=0)
    t = np.linspace(0, T-1, T, dtype='int')
#    plt.subplot(131)
#    # plt.plot(mean_norms, label=alg_name)
#    plt.plot(mean_regret.cumsum() / (np.arange(len(mean_regret)) + 1), label=alg_name)
#    plt.fill_between(t, low_quantile.cumsum() / (np.arange(len(mean_regret)) + 1),
#                     high_quantile.cumsum() / (np.arange(len(mean_regret)) + 1), alpha=0.15)
    plt.figure(1)
    print('mean_regret')
    print(alg_name, ' = ', mean_regret[-1])
    plt.fill_between(t, low_quantile, high_quantile, alpha = 0.15)
    plt.plot(mean_regret, label=alg_name)
    plt.title('d = {}'.format(model.n_features))
    plt.figure(2)
    print(alg_name, '= ', min(condition_satisfied.cumsum()))
    print('-'*100)
 #   plt.plot(condition_satisfied, label=alg_name)
    plt.title('d = {}'.format(model.n_features))
 #   plt.fill_between(t, low_quantile_condition, high_quantile_condition, alpha = 0.15)
    if alg_name != 'EfficientLinearBandit':
        plt.plot(condition_satisfied.cumsum()[:200], label=alg_name)
        plt.fill_between(t[:200], low_quantile_condition.cumsum()[:200], high_quantile_condition.cumsum()[:200], alpha = 0.15)

#ax = plt.subplot(131)
## plt.ylabel(r'$\|\hat{\theta} - \theta\|_{2}$')
#plt.ylabel(r'$R_t / t$')
#plt.xlabel("Rounds")
## # Turn off tick labels
## ax.set_yticklabels([])
## ax.set_xticklabels([])
#plt.legend()
#
#ax = plt.subplot(132)
#plt.ylabel("Cumulative Regret")
#plt.xlabel("Rounds")
## # Turn off tick labels
## ax.set_yticklabels([])
## ax.set_xticklabels([])
#plt.legend()
#
##ax = plt.subplot(223)
##plt.title('Model')
##plt.scatter(model.features[:, 0], model.features[:, 1])
##optimal_arm = np.argmax(means)
##plt.scatter([model.features[optimal_arm, 0]], [model.features[optimal_arm, 1]], color='red', label='Optimal arm')
##plt.scatter([model.features[baseline, 0]], [model.features[baseline, 1]], color='cyan', label='Baseline arm')
##plt.scatter([model.theta[0]], [model.theta[1]], color='yellow', label='Theta')
### # Turn off tick labels
### ax.set_yticklabels([])
### ax.set_xticklabels([])
##plt.legend()
#
#ax = plt.subplot(133)
#plt.ylabel("Margin")
#plt.xlabel("Rounds")

# # Turn off tick labels
# ax.set_yticklabels([])
# ax.set_xticklabels([])
plt.figure(1)
plt.legend()

#plt.savefig("model_random_{}_{}_seed_{}.png".format(alpha, model.n_actions, random_state))
plt.show()

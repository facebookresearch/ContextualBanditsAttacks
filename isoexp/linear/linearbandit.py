# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import cvxpy as cp
from scipy.optimize import minimize


class RandomArm(object):
    def __init__(self, initial_arms):
        self.arms = initial_arms

    def get_action(self):
        return np.random.choice(self.arms)

    def update(self, a_t, r_t):
        pass

    def reset(self):
        pass


class LinearBandit(object):
    def __init__(self, arm_features, reg_factor=1., delta=0.5,
                 bound_theta=None, noise_variance=None):
        self.arm_features = arm_features
        self.reg_factor = reg_factor
        self.delta = delta
        self.iteration = None
        self.bound_theta = bound_theta
        self.bound_features = np.max(np.sqrt(np.sum(np.abs(arm_features) ** 2, axis=1)))
        self.noise_variance = noise_variance

        self.reset()

    def reset(self):
        d = self.n_features
        self.A = self.reg_factor * np.eye(d, d)
        self.b = np.zeros((d,))

        self.range = 1
        self.est_bound_theta = 0
        self.est_bound_features = 0
        self.n_samples = 0
        self.iteration = 0

    @property
    def n_actions(self):
        return self.arm_features.shape[0]

    @property
    def n_features(self):
        return self.arm_features.shape[1]

    def auto_alpha(self):
        d = self.n_features
        return self.range * np.sqrt(d * np.log((1 + max(1, self.n_samples) / self.reg_factor) / self.delta)) \
               + np.sqrt(self.reg_factor) * np.linalg.norm(self.theta_hat, 2)

    def alpha(self, n_samples):
        d = self.n_features
        if self.bound_theta is None or self.noise_variance is None:
            # use estimated quantities
            sigma, B, D = self.range, self.est_bound_theta, self.bound_features
        else:
            sigma, B, D = self.noise_variance, self.bound_theta, self.bound_features
        return sigma * np.sqrt(d * np.log((1 + max(1, n_samples) * D * D / self.reg_factor) / self.delta)) \
               + np.sqrt(self.reg_factor) * B

    def get_action(self):
        self.iteration += 1

        # Let's not be biased with tiebreaks, but add in some random noise
        noise = np.random.random(self.n_actions) * 0.000001

        A_inv = np.linalg.inv(self.A)
        self.theta_hat = A_inv.dot(self.b)
        ta = np.diag(self.arm_features.dot(A_inv).dot(self.arm_features.T))

        sfactor = self.alpha(self.n_samples)
        ucb = self.arm_features.dot(self.theta_hat) + sfactor * ta

        ucb = ucb + noise

        choice = np.argmax(ucb)  # choose the highest
        # print(ucb, choice)
        return choice

    def update(self, a_t, r_t):
        # update the input vector
        phi = self.arm_features[a_t]
        self.A += np.outer(phi, phi)
        self.b += r_t * phi

        self.range = max(self.range, abs(r_t))
        self.est_bound_theta = np.linalg.norm(self.theta_hat)

        self.n_samples += 1


class EfficientLinearBandit(object):
    def __init__(self, arm_features, reg_factor=1., delta=0.5,
                 bound_theta=None, noise_variance=None):
        self.arm_features = arm_features
        self.reg_factor = reg_factor
        self.delta = delta
        self.iteration = None
        self.bound_theta = bound_theta
        self.bound_features = np.max(np.sqrt(np.sum(np.abs(arm_features) ** 2, axis=1)))
        self.noise_variance = noise_variance

        self.reset()

    def reset(self):
        d = self.n_features
        self.Ainv = np.eye(d, d) / self.reg_factor
        self.b = np.zeros((d,))
        self.range = 1
        self.est_bound_theta = 0
        self.est_bound_features = 0
        self.n_samples = 0
        self.iteration = 0

    @property
    def n_actions(self):
        return self.arm_features.shape[0]

    @property
    def n_features(self):
        return self.arm_features.shape[1]

    def auto_alpha(self):
        d = self.n_features
        sigma, B, D = self.noise_variance, self.bound_theta, self.bound_features
        return sigma * np.sqrt(d * np.log((1 + max(1, self.iteration - 1) * D * D / self.reg_factor) / self.delta)) \
               + np.sqrt(self.reg_factor) * B

    def alpha(self, n_samples):
        d = self.n_features
        if self.bound_theta is None or self.noise_variance is None:
            # use estimated quantities
            sigma, B, D = self.range, self.est_bound_theta, self.bound_features
        else:
            sigma, B, D = self.noise_variance, self.bound_theta, self.bound_features
        return sigma * np.sqrt(d * np.log((1 + max(1, n_samples) * D * D / self.reg_factor) / self.delta)) \
               + np.sqrt(self.reg_factor) * B

    def get_action(self, n_sam=None):
        self.iteration += 1

        if n_sam is None:
            n_sam = self.n_samples

        # Let's not be biased with tiebreaks, but add in some random noise
        noise = np.random.random(self.n_actions) * 0.000001

        # A_inv = np.linalg.inv(self.A)
        # assert np.allclose(A_inv, self.Ainv)
        self.theta_hat = np.dot(self.Ainv, self.b)
        ta = np.diag(np.dot(self.arm_features, np.dot(self.Ainv, self.arm_features.T)))

        sfactor = self.alpha(n_sam)
        ucb = self.arm_features.dot(self.theta_hat) + sfactor * np.sqrt(ta)

        ucb = ucb + noise

        choice = np.argmax(ucb)  # choose the highest
        # print(ucb, choice)
        return choice

    def update(self, a_t, r_t):
        # update the input vector
        phi = self.arm_features[a_t]
        # self.A += np.outer(phi, phi)
        self.Ainv = self.Ainv - np.dot(self.Ainv, np.dot(np.outer(phi, phi), self.Ainv)) / (
                    1. + np.dot(phi.T, np.dot(self.Ainv, phi)))
        self.b += r_t * phi

        self.range = max(self.range, abs(r_t))
        # self.est_bound_theta = np.linalg.norm(self.theta_hat)

        self.n_samples += 1

        
class UCB_GLM() :
    
    def __init__(self, arm_features, reg_factor = 1, delta = 0.1, 
                 bound_theta = 1, 
                 link_function = lambda x : x, 
                 noise_variance = None,
                 model = None,
                 conservative_level=0.1, 
                 tighter_ucb = False,
                 kappa = None) :
        
        self.conservative_level = conservative_level
        self.tighter_ucb = tighter_ucb
        self.arm_features = arm_features
        self.reg_factor = reg_factor
        self.delta = delta
        self.bound_theta = bound_theta
        self.model = model
        self.n_actions, self.d = arm_features.shape
        self.noise_variance = noise_variance
        
        if self.model == 'gaussian' :
            self.link_function = lambda x : x
            self.kappa = 1
            self.L = 1
        elif self.model == 'bernoulli' :
            self.link_function = lambda x : 1/(1+np.exp(-x))
            if kappa is None :
                self.kappa = 1/1000
            else :
                self.kappa = kappa
            self.L = 1/4

        self.reset()

    
    def reset(self) :
        self.rewards_history = []
        self.features_history = []
        self.A = self.reg_factor * np.eye(self.d, self.d)/self.kappa
        self.Ainv = np.eye(self.d, self.d)*self.kappa / self.reg_factor
        self.n_samples = 0
        self.iteration = 0
        self.theta_hat = np.zeros(self.d)
            
    def solve_MLE(self, rewards_history, features_history) :
        
        if self.iteration > 1:
            if not self.model is None :
                n_samples = len(self.rewards_history)
                n_features = self.d
                X = np.zeros((n_samples, n_features))
                X = 1*np.array(self.features_history)
                y = (np.array(self.rewards_history).reshape((n_samples,)))
                beta = cp.Variable(n_features)
                lambd = cp.Parameter(nonneg = True)
                lambd.value = self.reg_factor/2
                
                if self.model == 'bernoulli' :
                    
                    log_likelihood = cp.sum(cp.multiply(y, X @ beta) -
                            cp.log_sum_exp(cp.vstack([np.zeros(n_samples), X @ beta]), axis=0)) - lambd * cp.norm(beta, 2)
                    problem = cp.Problem(cp.Maximize(log_likelihood))
                    problem.solve(verbose = False, warm_start = False, max_iters = 200)
                    return beta.value
                else :
                    log_likelihood = cp.sum( cp.multiply(y, X @ beta) -
                            cp.power(X@beta, 2)/2) - lambd * cp.norm(beta, 2)
                    problem = cp.Problem(cp.Maximize(log_likelihood))
                    problem.solve(verbose = False, warm_start = False, max_iters = 200)
                    return beta.value
        else :
            return np.zeros((self.d,))
                
    def auto_alpha(self, tight_bound):
        if tight_bound :
            return np.sqrt(2*self.L*np.log(self.n_samples + 1)/self.kappa)
        else :
            sigma, B = self.noise_variance, self.bound_theta
            return np.sqrt(self.reg_factor/self.kappa)*B + sigma*np.sqrt( self.d*np.log(1 + self.iteration*self.kappa/(self.reg_factor*self.d)) + 2*np.log(1/self.delta))/self.kappa
 
                   
    def get_action(self) :
        self.iteration += 1
        
        noise = np.random.random(self.n_actions) * 0.0000001
        
        self.theta_hat = self.solve_MLE(self.rewards_history, self.features_history)
        beta = self.auto_alpha(self.tighter_ucb)
        ta = np.diag(np.dot(self.arm_features, np.dot(self.Ainv, self.arm_features.T)))
        ucb = self.arm_features.dot(self.theta_hat) + beta * ta
        ucb = ucb + noise
        UCB_action= np.argmax(ucb) 
        return UCB_action
    
    def update(self, a_t, r_t):
        
        phi = self.arm_features[a_t]
        self.Ainv = self.Ainv - np.dot(self.Ainv, np.dot(np.outer(phi, phi), self.Ainv)) / (1. + np.dot(phi.T, np.dot(self.Ainv, phi)))
        self.A += np.outer(phi,phi)
        self.rewards_history.append(r_t)
        self.features_history.append(phi)
        self.n_samples += 1
    
    def check_condition(self, theta):
        
        temp  = np.array(self.rewards_history).reshape((len(self.rewards_history),)) - self.link_function(np.array(self.features_history).dot(self.theta_hat))
        temp  = temp*np.array(self.features_history).reshape((self.d,len(self.rewards_history)))
        temp = temp.T
        temp = np.sum(temp, axis = 0) - self.reg_factor*theta
        return temp



class LinPHE():

    def __init__(self, arm_features, reg_factor=1, alpha=2):
        self.arm_features = arm_features
        self.reg_factor = reg_factor
        self.alpha = alpha
        self.reset()

    def reset(self):
        self.K, self.d = self.arm_features.shape
        self.design_matrix = self.reg_factor * np.eye(self.d)
        self.inv_design_matrix = np.eye(self.d) / (self.reg_factor)
        self.iteration = 0

        self.N = np.zeros((self.K,))
        self.S = np.zeros((self.K,))

    def get_action(self):
        if self.iteration < -1: #self.d:
            choice = np.random.randint(self.K)
        else:
            temp = np.zeros((self.d,))
            for j in range(self.K):
                Z = np.random.binomial(1 + int(self.alpha * self.N[j]), 0.5)
                temp = temp + self.arm_features[j] * (self.S[j] + Z)
            self.theta_hat = np.dot(self.inv_design_matrix, temp) / (self.alpha + 1)
            ucb = self.arm_features.dot(self.theta_hat)
            noise = np.random.randn(self.K) * 10 ** -7
            ucb = ucb + noise
            choice = np.argmax(ucb)
        self.iteration += 1
        return choice

    def update(self, a_t, r_t):
        self.S[a_t] += r_t * 1
        self.N[a_t] += 1
        x = self.arm_features[a_t]
        self.design_matrix = self.design_matrix + np.outer(x, x)
        self.inv_design_matrix = (self.inv_design_matrix - np.dot(self.inv_design_matrix,
                                                                  np.dot(np.outer(x, x), self.inv_design_matrix)) / (
                                              1. + np.dot(x.T, np.dot(self.inv_design_matrix, x))))


if __name__ == '__main__':
    import sys

    sys.path[0] = '/Users/evrard/Desktop/monotone_mabs/'
    # from isoexp.linear.linearbandit import EfficientLinearBandit, LinearBandit
    # from isoexp.conservative.linearmabs import EfficientConservativeLinearBandit, NewCLUB, SafetySetCLUCB, BatchedCLUCB, LinearOracle, LinUCBOracle
    from isoexp.linear.linearmab_models import RandomLinearArms, RandomLogArms
    from tqdm import trange
    from collections import namedtuple
    from joblib import Parallel, delayed

    seed = np.random.randint(0, 10 ** 5)
    MABResults = namedtuple('MABResults', 'regret,norm_error')
    noise = 0.1
    model = RandomLogArms(n_actions=20, n_features=2, noise=noise,
                             bound_features=1,
                             random_state=seed)
    model.features = model.features
    theta_bound = np.linalg.norm(model.theta, 2)
    link = lambda x: 1 / (1 + np.exp(-x))
    link_means = np.array([link(np.dot(model.theta, el)) for el in model.features])
    means = np.array([(np.dot(model.theta, el)) for el in model.features])
    T = 1500
    PARALLEL = True
    nb_simu = 10

    
    algorithms = {
        #    'EfficientLinearBandit': EfficientLinearBandit(arm_features=model.features,
        #                                     reg_factor=1.,
        #                                     delta=0.1,
        #                                     noise_variance=noise,
        #                                     bound_theta=theta_bound),
        'UCB-GLM-tight-bound': UCB_GLM(arm_features=model.features,
                                       bound_theta=theta_bound,
                                       model='bernoulli',
                                       noise_variance=noise,
                                       reg_factor=1,
                                       delta=0.1,
                                       tighter_ucb=True),
        'UCB-GLM': UCB_GLM(arm_features=model.features,
                           bound_theta=theta_bound,
                           model='bernoulli',
                           noise_variance=noise,
                           reg_factor=1,
                           delta=0.1,
                           tighter_ucb=False)}

    if PARALLEL:
        import multiprocessing

        num_cores = multiprocessing.cpu_count()


        def work(alg_name, alg):

            regret = np.zeros((nb_simu, T))
            norms = np.zeros((nb_simu, T))

            for k in trange(nb_simu, desc='Simulating {}'.format(alg_name)):

                alg.reset()

                for t in trange(T, desc='Current episode :', leave=True):
                    a_t = alg.get_action()
                    # print(a_t)
                    r_t = model.reward(a_t)
                    alg.update(a_t, r_t)

                    regret[k, t] = link(model.best_arm_reward()) - link(np.dot(model.theta, model.features[a_t]))
                    if hasattr(alg, 'theta_hat'):
                        norms[k, t] = np.linalg.norm(alg.theta_hat - model.theta, 2)

            return alg_name, MABResults(regret=regret, norm_error=norms)


        results = Parallel(n_jobs=num_cores, verbose=1)(
            delayed(work)(alg_name, algorithms[alg_name]) for alg_name in algorithms.keys())
    else:
        results = []
        for alg_name, alg in algorithms.items():
            regret = np.zeros((nb_simu, T))
            norms = np.zeros((nb_simu, T))
            cond = np.zeros((nb_simu, T))
            draws = np.zeros((nb_simu, T))
            for k in trange(nb_simu, desc='Simulating {}'.format(alg_name)):

                alg.reset()

                for t in trange(T, desc='Current episode ', leave=True):
                    a_t = alg.get_action()
                    r_t = model.reward(a_t)
                    alg.update(a_t, r_t)
                    regret[k, t] = link(model.best_arm_reward()) - link(np.dot(model.theta, model.features[a_t]))

                    if hasattr(alg, 'theta_hat'):
                        norms[k, t] = np.linalg.norm(alg.theta_hat - model.theta, 2)
                    draws[k, t] = a_t

            results += [(alg_name, MABResults(regret=regret, norm_error=norms))]
    import pylab as plt

    for (alg_name, val) in results:
        mean_regret = np.mean(val.regret.cumsum(axis=0), axis=0)
        mean_norms = np.mean(val.norm_error, axis=0)
        t = np.linspace(1, T + 1, T, dtype='int')
        low_quantile = np.quantile(val.regret/t, 0.1, axis=0)
        high_quantile = np.quantile(val.regret/t, 0.9, axis=0)
        plt.figure(0)
        plt.semilogx(mean_regret.cumsum()/t, label=alg_name)
        plt.legend()
        # plt.fill_between(t, low_quantile, high_quantile, alpha = 0.15)
        plt.figure(1)
        plt.plot(mean_norms, label=alg_name)
        plt.legend()
    plt.show()

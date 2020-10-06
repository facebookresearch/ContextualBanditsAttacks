# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import numpy as np
import sys
import numpy.random as npr
import cvxpy as cp
from tqdm import trange
from tqdm import tqdm


def UCB1(T, MAB, alpha=1.):
    """
    Args:
        T (int): horizon
        MAB (list): list of available MAB models
        alpha (float): shrink confidence interval
    Returns:
        rewards (array-like): observed rewards
        draws (array-like): indexes of selected arms
    """
    K = len(MAB)
    rewards = np.zeros((T,))
    draws = np.zeros((T,))
    N = np.ones((K,))  # number of observations of each arm
    S = np.zeros((K,))  # sum of rewards for each arm

    for k in range(K):
        a = k
        r = MAB[a].sample()

        # update quantities
        rewards[k] = r
        draws[k] = a
        S[a] += r
        N[a] += 1

    for t in range(K, T):
        # select the arm
        ucb = S / N + alpha * np.sqrt(np.log(t + 1) / N)

        idxs = np.flatnonzero(np.isclose(ucb, ucb.max()))
        a = np.asscalar(np.random.choice(idxs))

        r = MAB[a].sample()

        # update quantities
        rewards[t] = r
        draws[t] = a
        S[a] += r
        N[a] += 1

    return rewards, draws


def TS(T, MAB):
    """
    Args:
        T (int): horizon
        MAB (list): list of available MAB models
    Returns:
        rewards (array-like): observed rewards
        draws (array-like): indexes of selected arms
    """
    K = len(MAB)
    rewards = np.zeros((T,))
    draws = np.zeros((T,))

    alphas = np.ones((K,))
    betas = np.ones((K,))

    for t in range(T):
        # sample the arm
        thetas = np.random.beta(alphas, betas)

        # select and apply action
        a = np.argmax(thetas)
        r = MAB[a].sample()

        # update distribution
        alphas[a] += r
        betas[a] += 1 - r

        rewards[t] = r
        draws[t] = a

    return rewards, draws


def epsGREEDY(T, MAB, epsilon=0.1):
    """
    Args:
        T (int): horizon
        MAB (list): list of available MAB models
    Returns:
        rewards (array-like): observed rewards
        draws (array-like): indexes of selected arms
    """
    K = len(MAB)
    rewards = np.zeros((T,))
    draws = np.zeros((T,))

    N = np.ones((K,))  # number of observations of each arm
    S = np.zeros((K,))  # sum of rewards for each arm

    for k in range(K):
        a = k
        r = MAB[a].sample()

        # update quantities
        rewards[k] = r
        draws[k] = a
        S[a] += r
        N[a] += 1

    for t in range(K, T):
        # select the arm
        ucb = S / N

        rnd = np.random.rand()
        if rnd <= epsilon:
            a = np.random.choice(K)
        else:
            idxs = np.flatnonzero(np.isclose(ucb, ucb.max()))
            a = np.asscalar(np.random.choice(idxs))

        r = MAB[a].sample()

        # update quantities
        rewards[t] = r
        draws[t] = a
        S[a] += r
        N[a] += 1

    return rewards, draws


def SoftMAB(T, MAB, temp=1.0):
    """
    Args:
        T (int): horizon
        MAB (list): list of available MAB models
    Returns:
        rewards (array-like): observed rewards
        draws (array-like): indexes of selected arms
    """
    K = len(MAB)
    rewards = np.zeros((T,))
    draws = np.zeros((T,))

    N = np.ones((K,))  # number of observations of each arm
    S = np.zeros((K,))  # sum of rewards for each arm

    for k in range(K):
        a = k
        r = MAB[a].sample()

        # update quantities
        rewards[k] = r
        draws[k] = a
        S[a] += r
        N[a] += 1

    for t in range(K, T):
        # select the arm
        ucb = S / N

        proba = np.exp(ucb / temp)
        proba = proba / np.sum(proba)
        a = np.random.choice(K, p=proba)

        r = MAB[a].sample()

        # update quantities
        rewards[t] = r
        draws[t] = a
        S[a] += r
        N[a] += 1

    return rewards, draws


def ExploreThenExploit(T, MAB, T1):
    K = len(MAB)
    rewards = np.zeros((T,))
    draws = np.zeros((T,))

    N = np.ones((K,))  # number of observations of each arm
    S = np.zeros((K,))  # sum of rewards for each arm

    T1 = np.ceil(T1).astype(np.int)

    for t in range(T1):
        a = np.random.choice(K)
        r = MAB[a].sample()

        # update quantities
        rewards[t] = r
        draws[t] = a
        S[a] += r
        N[a] += 1

    for t in range(T1, T):
        # select the arm
        ucb = S / N
        idxs = np.flatnonzero(np.isclose(ucb, ucb.max()))
        a = np.asscalar(np.random.choice(idxs))
        r = MAB[a].sample()

        # update quantities
        rewards[t] = r
        draws[t] = a
        S[a] += r
        N[a] += 1

    return rewards, draws


def UCBV(T, MAB, alpha=1.):
    """
    Args:
        T (int): horizon
        MAB (list): list of available MAB models
        alpha (float): shrink confidence interval
    Returns:
        rewards (array-like): observed rewards
        draws (array-like): indexes of selected arms
    """
    K = len(MAB)
    rewards = np.zeros((T,))
    draws = np.zeros((T,))

    N = np.ones((K,))   # number of observations of each arm
    S = np.zeros((K,))  # sum of rewards for each arm
    M = np.zeros((K,))  # second moment (for Welford's algorithm)

    vars = np.ones((K,)) * np.inf
    for t in range(T):
        # select the arm
        ln = np.log(t + 1)
        ucb = S / N + alpha * (np.sqrt(vars * ln / N) + ln / N)
        ucb[N < 2] = sys.maxsize

        idxs = np.flatnonzero(np.isclose(ucb, ucb.max()))
        a = np.asscalar(np.random.choice(idxs))

        r = MAB[a].sample()

        # update quantities
        rewards[t] = r
        draws[t] = a
        old_mean = S[a] / N[a] if N[a] > 0 else 0
        S[a] += r
        N[a] += 1
        M[a] = M[a] + (r - old_mean) * (r - S[a]/N[a])
        vars[a] = M[a] / N[a]  # update variance estimate

    return rewards, draws

def BootstrapedUCB(T, MAB, delta = 0.1, b_rep = 200):

    K = len(MAB)
    rewards = np.zeros((T,))
    draws = np.zeros((T,))

    N = np.zeros((K,))  
    S = np.zeros((K,))
    rewards_arm = {}
    for k in range(K):
        a = k
        r = 1*MAB[a].sample().squeeze()

        rewards[k] = r
        draws[k] = a
        rewards_arm[k] = [r]
        S[a] += r
        N[a] += 1

    for t in range(K, T):
        alpha = 1/(t+1)
        bootstrap_quantile = quantile((1-delta)*alpha, S, N, rewards_arm, B = b_rep)
        phi = np.sqrt(2*np.log(1/alpha)/N)
        ## Theoretical ucb 
        #ucb = S / N   + (bootstrap_quantile + np.sqrt(np.log(2/(delta*alpha))/N)*phi)
        ## Ucb used in practice 
        ucb = S / N   + (bootstrap_quantile + np.sqrt(1/N)*phi)
        idxs = np.flatnonzero(np.isclose(ucb, ucb.max()))
        a = np.asscalar(np.random.choice(idxs))
        r = 1*MAB[a].sample().squeeze()
        rewards[t] = r
        draws[t] = a
        rewards_arm[a].append(r)
        S[a] += r
        N[a] += 1

    return rewards, draws

def quantile(alpha, S, N, rwds, B = 100, distrib = 'rademacher') : 
    
    means = np.nan_to_num(S/N)
    K = len(N)
    np_quantile = np.zeros(K)
    for k in range(K) :
        n = N[k]
        if n > 0 :
            bootstrap_avg = np.zeros(B)
            if distrib == 'rademacher' : 
                weights = 2*npr.binomial(1, 1/2, size = (int(B),int(n))) - 1
            elif distrib =='gaussian' :
                weights = npr.randn(int(B),int(n))
            history = np.array(rwds[k]) - means[k]
            bootstrap_avg = (np.dot(weights, history)/n)
            np_quantile[k] = np.percentile(bootstrap_avg, 100*(1 - alpha), interpolation = 'nearest')
        else :
            np_quantile[k] = +np.inf
    return np_quantile


def PHE(T, MAB, alpha = 2) :
    K = len(MAB)
    rewards = np.zeros((T,))
    draws = 0*rewards
    N = np.zeros((K,))
    S = np.zeros((K,))
    biased_test = np.zeros((K,))
    
    for k in range(K):
        a = k
        r = 1*MAB[a].sample().squeeze()
        rewards[k] = r
        draws[k] = a
        S[a] +=r
        N[a] +=1
        
    for t in range(K, T) :
        
        for i in range(K) :
            Z = np.random.binomial(1,1/2, size = int(alpha*N[i]))
            biased_test[i] = (np.sum(Z) + S[i])/((alpha+1)*N[i])
        idxs = np.flatnonzero(np.isclose(biased_test, biased_test.max()))
        a = np.asscalar(np.random.choice(idxs))
        r = 1*MAB[a].sample().squeeze()  
        N[a] +=1
        S[a] +=r
        rewards[t] = r
        draws[t] = a
    return rewards, draws  

def Random_exploration(T, MAB, alpha = 2) :
    K = len(MAB)
    rewards = np.zeros((T,))
    draws = 0*rewards
    N = np.zeros((K,))
    S = np.zeros((K,))
    biased_test = np.zeros((K,))
    for k in range(K):
        
        a = k
        r = 1*MAB[a].sample().squeeze()
        
        rewards[k] = r
        draws[k] = a
        S[a] +=r
        N[a] +=1
    for t in range(K, T) :
        for i in range(K) :
            Z = np.random.binomial(1,1/2, size = int(alpha*N[i]))
            biased_test[i] = np.nan_to_num(np.mean(Z))+ S[i]/N[i]
        idxs = np.flatnonzero(np.isclose(biased_test, biased_test.max()))

        a = np.asscalar(np.random.choice(idxs))
        
        r = 1*MAB[a].sample().squeeze()  
        N[a] +=1
        S[a] +=r
        rewards[t] = r
        draws[t] = a
    return rewards, draws  

def EXP3_IX(T, MAB, eta = 0.1, gamma = 0):

    K = len(MAB)
    losses = np.zeros((K,))
    rewards = np.zeros((T,))
    draws = 0*rewards
    sum_exp = K
    exp_losses = np.ones((K,))
    arms = np.linspace(0, K-1, K, dtype='int')
    for t in range(T):
        # print('cum_losses =', exp_losses)
        # print('sum losses=', sum_exp)
        P = exp_losses/sum_exp
        # print('P =', P)
        action = np.random.choice(arms, p=P)
        X = 1*MAB[action].sample().squeeze()
        losses[action] = losses[action] + (1 - X)/(gamma + P[action])
        exp_losses[action] = exp_losses[action]*np.exp(-eta* (1 - X)/(gamma + P[action]))
        sum_exp = np.sum(exp_losses)
        rewards[t] = X
        draws[t] = action

    return rewards, draws

def attacked_EXP3_IX(T, MAB, target_arm, eta = None, gamma = None, delta=0.99):

    K = len(MAB)
    losses = np.zeros((K,))
    rewards = np.zeros((T,))
    draws = 0*rewards
    sum_exp = K
    exp_losses = np.ones((K,))
    arms = np.linspace(0, K-1, K, dtype='int')
    N = np.ones((K,))  # number of observations of each arm
    S = np.zeros((K,))
    beta = np.zeros((K,))
    attacks = np.zeros((T,))
    time_of_attacks = np.zeros((T,))
    if eta is None or gamma is None:
        eta = np.sqrt(2*np.log(K + 1)/(K*T))
        gamma = np.sqrt(2*np.log(K + 1)/(K*T))/2

    for t in range(T):
        P = exp_losses/sum_exp
        if t < K:
            action = t
            attack_t = 0
        else:
            time_of_attacks[t] = 1
            action = np.random.choice(arms, p=P)
            if action != target_arm:
                beta = np.sqrt(np.log(np.pi ** 2 * K * N ** 2 / (3 * delta)) / (2*N))
                attack_t = - np.maximum((S / N)[action] - (S / N)[target_arm] + beta[action] + beta[target_arm], 0)
            else:
                attack_t = 0
        attacks[t] = attack_t
        true_X = 1*MAB[action].sample().squeeze()
        X = true_X + attack_t
        losses[action] = losses[action] + (1 - X)/(gamma + P[action])
        exp_losses[action] = exp_losses[action]*np.exp(-eta*(1 - X)/(gamma + P[action]))
        sum_exp = np.sum(exp_losses)
        rewards[t] = true_X
        draws[t] = action
        N[action] += 1
        S[action] += true_X

    return rewards, draws, attacks, time_of_attacks


def attacked_UCB1(T, MAB, target_arm, alpha=1., delta=0.99, constant_attack=False):

    K = len(MAB)
    rewards = np.zeros((T,))
    draws = np.zeros((T,))
    N = np.ones((K,))  # number of observations of each arm
    S = np.zeros((K,))  # sum of rewards for each arm
    N_pre = np.ones((K,))  # number of observations of each arm
    S_pre = np.zeros((K,))
    attacks = np.zeros((T,))
    time_of_attacks = np.zeros((T,))

    for k in range(K):
        a = k
        r = MAB[a].sample()
        rewards[k] = r
        draws[k] = a
        S[a] += r
        N[a] += 1
        S_pre[a] += r
        N_pre[a] += 1
        attacks[k] = 0

    for t in range(K, T):
        # select the arm
        ucb = S / N + alpha * np.sqrt(np.log(t + 1) / N)
        beta = np.sqrt(np.log(np.pi**2*K*N**2/(3*delta))/(2*N))
        idxs = np.flatnonzero(np.isclose(ucb, ucb.max()))
        a = np.asscalar(np.random.choice(idxs))
        if a != target_arm:
            time_of_attacks[t] = 1
            if constant_attack:
                attack_t = - 2 * np.maximum(0, MAB[a].mean - MAB[target_arm].mean)
            else:
                beta = np.sqrt(np.log(np.pi ** 2 * K * N ** 2 / (3 * delta)) / (2 * N))
                attack_t = - np.maximum((S_pre / N)[a] - (S_pre / N)[target_arm] + beta[a] + beta[target_arm], 0)
        else:
            attack_t = 0
        attacks[t] = attack_t
        r = MAB[a].sample()
        false_r = r + attack_t
        # update quantities
        rewards[t] = r
        draws[t] = a
        S[a] += false_r
        N[a] += 1
        S_pre[a] += r
        N_pre[a] += 1

    return rewards, draws, attacks, time_of_attacks

def EXP3_P(T, MAB, eta=0.1, gamma=0):

    K = len(MAB)
    S = np.zeros((K,))
    rewards = np.zeros((T,))
    draws = 0*rewards
    sum_exp = K
    exp_S = np.ones((K,))
    arms = np.linspace(0, K-1, K, dtype='int')
    for t in range(T):
        P = (1 - gamma) * exp_S / sum_exp + gamma / K * np.ones((K,))
        if t < K:
            action = t
            attack_t = 0
        else:
            # print('Probability distribution:', P)
            action = np.random.choice(arms, p=P)
        X = 1*MAB[action].sample().squeeze()
        S = S + 1
        S[action] = S[action] - (1 - X)/P[action]
        exp_S = exp_S*np.exp(eta)
        exp_S[action] = exp_S[action]*np.exp(-eta *(1 - X)/P[action])
        sum_exp = np.sum(exp_S)
        rewards[t] = X
        draws[t] = action
    return rewards, draws

def attacked_EXP3_P(T, MAB, target_arm, eta = None, gamma = None, delta=0.99, constant_attack=False):

    K = len(MAB)
    estimated_S = np.zeros((K,))
    rewards = np.zeros((T,))
    draws = 0 * rewards
    sum_exp = K
    exp_estimated_S = np.ones((K,))
    arms = np.linspace(0, K - 1, K, dtype='int')
    N = np.ones((K,))  # number of observations of each arm
    S = np.zeros((K,))
    beta = np.zeros((K,))
    attacks = np.zeros((T,))
    time_of_attacks = np.zeros((T,))
    if eta is None and gamma is None:
        eta = np.sqrt(np.log(K + 1) / (K * T))
        gamma = 0
    elif eta is None:
        eta = np.sqrt(np.log(K + 1) / (K * T))
    elif gamma is None:
        gamma = 0
    for t in range(T):
        P = (1 - gamma) * exp_estimated_S / sum_exp + gamma/K*np.ones((K,))
        if t < K:
            action = t
            attack_t = 0
        else:
            action = np.random.choice(arms, p=P)
            if action != target_arm:
                time_of_attacks[t] = 1
                if constant_attack:
                    attack_t = - 2*np.maximum(0, MAB[action].mean - MAB[target_arm].mean)
                else:
                    beta = np.sqrt(np.log(np.pi ** 2 * K * N ** 2 / (3 * delta)) / (2 * N))
                    attack_t = - np.maximum((S / N)[action] - (S / N)[target_arm] + beta[action] + beta[target_arm], 0)
            else:
                attack_t = 0
        attacks[t] = attack_t
        true_X = 1 * MAB[action].sample().squeeze()
        X = true_X + attack_t
        estimated_S = estimated_S + 1
        estimated_S[action] = estimated_S[action] - (1 - X) /P[action]
        exp_estimated_S = exp_estimated_S*np.exp(eta)
        exp_estimated_S[action] = exp_estimated_S[action] * np.exp(eta * (- (1 - X) /P[action]))
        sum_exp = np.sum(exp_estimated_S)
        rewards[t] = true_X
        draws[t] = action
        N[action] += 1
        S[action] += true_X

    return rewards, draws, attacks, time_of_attacks

def FTRL(T, MAB, eta=10, alg='exp_3'):

    K = len(MAB)
    S = np.zeros((K,))
    losses = np.zeros((K,))
    rewards = np.zeros((T,))
    draws = 0*rewards
    arms = np.linspace(0, K-1, K, dtype='int')

    for t in trange(T):
        x = cp.Variable(K, pos=True)
        temp_1 = cp.Constant(value=np.ones((K,)))
        temp_2 = cp.Constant(value=losses)
        constraints = [cp.sum(cp.multiply(temp_1, x)) == 1]
        if alg == 'log_barrier':
            obj = cp.Minimize(cp.sum(cp.multiply(temp_2, x)) - 1/eta*cp.sum(cp.log(x)))
        elif alg == 'inf':
            obj = cp.Minimize(cp.sum(cp.multiply(temp_2, x)) - 2/eta*cp.sum(cp.sqrt(x)))
        else:
            obj = cp.Minimize(cp.sum(cp.multiply(temp_2, x)) + 1/eta*(cp.sum(cp.kl_div(x, temp_1)) - K))
        pb = cp.Problem(obj, constraints)
        try:
            pb.solve()
            P = x.value
        except:
            P = np.ones((K,))/K
        # print('Probability distribution:', P)
        if not np.sum(P) == 1:
            P = P/np.sum(P)
        action = np.random.choice(arms, p=P)
        X = 1*MAB[action].sample().squeeze()
        S[action] = S[action] + X/P[action]
        losses[action] = losses[action] + (-X)/P[action]
        rewards[t] = X
        draws[t] = action
    return rewards, draws


def attacked_FTRL(T, MAB, target_arm, eta=10, alg='exp_3', delta=0.99, constant_attack=False):

    K = len(MAB)
    true_S = np.zeros((K,))
    true_losses = np.zeros((K,))
    N = np.zeros((K,))
    estimated_losses = np.zeros((K,))
    rewards = np.zeros((T,))
    draws = 0*rewards
    arms = np.linspace(0, K-1, K, dtype='int')
    attacks = np.zeros((T,))
    time_of_attacks = np.zeros((T,))
    for t in trange(T):
        x = cp.Variable(K, pos=True)
        temp_1 = cp.Constant(value=np.ones((K,)))
        temp_2 = cp.Constant(value=estimated_losses)
        constraints = [cp.sum(cp.multiply(temp_1, x)) == 1]
        if alg == 'log_barrier':
            obj = cp.Minimize(cp.sum(cp.multiply(temp_2, x)) - 1/eta*cp.sum(cp.log(x)))
        elif alg == 'inf':
            obj = cp.Minimize(cp.sum(cp.multiply(temp_2, x)) - 2/eta*cp.sum(cp.sqrt(x)))
        else:
            obj = cp.Minimize(cp.sum(cp.multiply(temp_2, x)) + 1/eta*(cp.sum(cp.kl_div(x, temp_1)) - K))
        pb = cp.Problem(obj, constraints)
        try:
            pb.solve()
            P = x.value
        except:
            P = np.ones((K,))/K
        # print("\nThe optimal value is", pb.value)
        # print("A solution x is")
        # print(x.value)
        # print("A dual solution corresponding to the inequality constraints is")
        # print(pb.constraints[0].dual_value)
        # print('Probability distribution:', P)
        if not np.sum(P) == 1:
            P = P/np.sum(P)
        if t < K:
            action = t
            attack_t = 0
        else:
            action = np.random.choice(arms, p=P)
            if action != target_arm:
                time_of_attacks[t] = 1
                beta = np.sqrt(np.log(np.pi ** 2 * K * N ** 2 / (3 * delta)) / (2 * N))
                if constant_attack:
                    attack_t = - 2*np.maximum(0, MAB[action].mean - MAB[target_arm].mean)
                else:
                    attack_t = - np.maximum((true_S / N)[action] - (true_S / N)[target_arm] + beta[action]
                                            + beta[target_arm], 0)
            else:
                attack_t = 0
        attacks[t] = attack_t
        true_X = 1*MAB[action].sample().squeeze()
        X = true_X + attack_t
        true_S[action] = true_S[action] + true_X
        true_losses[action] = true_losses[action] + (1-true_X)/P[action]
        estimated_losses[action] = estimated_losses[action] + (1 - X)/P[action]
        N[action] = N[action] + 1
        rewards[t] = true_X
        draws[t] = action
    return rewards, draws, attacks, time_of_attacks

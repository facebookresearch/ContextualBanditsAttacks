# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


def in_hull(points, x):
    n_points = len(points)
    n_dim = len(x)
    c = np.zeros(n_points)
    A = np.r_[points.T, np.ones((1, n_points))]
    b = np.r_[x, np.ones(1)]
    lp = linprog(c, A_eq=A, b_eq=b)
    return lp.success

def compute_relaxed_attack(alg, a_star, x_star, slack=10**-10):
    d = model.n_features
    delta = cp.Variable(d)
    obj = cp.Minimize(cp.quad_form(delta, np.eye(d))/2)
    theta_a_star = alg.thetas_hat[a_star]
    betas = alg.alpha()
    constraints = []
    P_a_star = sqrtm(alg.inv_design_matrices[a_star])
    for a in range(len(alg.thetas_hat)):
        if a != a_star:
            theta = alg.thetas_hat[a]
            P = sqrtm(alg.inv_design_matrices[a])
            temp_constraint = (theta - theta_a_star)@(x_star+delta) + betas[a]*cp.norm(P@(x_star+delta))\
                              #- betas[a_star]R * (cp.norm(P_a_star @ x_star) + (alg.inv_design_matrices[a] @ x_star) @
                              #                  delta/cp.norm(P_a_star @ x_star))
            constraints.append(temp_constraint + slack <= 0)
    prob = cp.Problem(obj, constraints)
    try:
        prob.solve(verbose=False, max_iters=1000, feastol=10**-8)
        epsilon = delta.value
        # print('epsilon =', epsilon)
        # for a in range(len(alg.thetas_hat)):
        #     if a != a_star:
        #         theta_a_star = alg.thetas_hat[a_star]
        #         betas = alg.alpha()
        #         theta = alg.thetas_hat[a]
        #         P = sqrtm(alg.inv_design_matrices[a])
        #         P_a_star = sqrtm(alg.inv_design_matrices[a_star])
        #         print('The constraint for arm {}:'.format(a), np.dot(theta - theta_a_star, (x_star+epsilon)) + betas[a]*np.linalg.norm(P.dot((x_star+epsilon))) \
        #                       - betas[a_star] * (np.linalg.norm(P_a_star.dot(x_star)) +
        #                                          np.dot((alg.inv_design_matrices[a].dot(x_star)), epsilon)/np.linalg.norm(P_a_star.dot(x_star))))

    except:
        print('Exception')
        epsilon = np.zeros((d,))
    # print('epsilon =', epsilon)
    if epsilon is None:
        return np.zeros((d,))
    return epsilon

def compute_attack(alg, a_star, x_star, slack=10 **-10):
    d = model.n_features
    func = lambda delta: np.linalg.norm(delta)/2
    theta_a_star = alg.thetas_hat[a_star]
    P_a_star = sqrtm(alg.inv_design_matrices[a_star])
    betas = alg.alpha()
    constraints_list = []
    for a in range(len(alg.thetas_hat)):
        if a != a_star:
            theta = alg.thetas_hat[a]
            P = sqrtm(alg.inv_design_matrices[a])
            temp_constraint = lambda delta, P=P, P_a_star=P_a_star, theta=theta, theta_a_star=theta_a_star, beta=betas[a], beta_a_star=betas[a_star]: \
                -((theta - theta_a_star).dot((x_star + delta)) + beta * np.linalg.norm(P.dot((x_star + delta)))
                  - beta_a_star * np.linalg.norm(P_a_star.dot((x_star + delta))) + slack)
            temp_cons = {'type': 'ineq', 'fun': temp_constraint}
            constraints_list.append(temp_cons)
    cons = tuple(constraints_list)
    res = minimize(func, -x_star, method='SLSQP', constraints=cons)
    # print(res.message)
    try:
        epsilon = res.x
    except:
        epsilon = np.zeros((d,))
    if epsilon is None:
        return np.zeros((d,))
    return epsilon

n = 10  # Number of colors
new_colors = [plt.get_cmap('Set1')(1. * i / n) for i in range(n)]
# linestyle_cycler = cycler('linestyle', ['-', '--', ':', '-.', '-', '--', ':', '-.', '-'])
# plt.rc('axes', prop_cycle=(cycler('color', new_colors) + linestyle_cycler))
plt.rc('lines', linewidth=2)

MABResults = namedtuple('MABResults', 'regret, attack_cond, target_draws, thetas, design_matrix, context_norm, betas')
seed = np.random.randint(0, 10 ** 5)
print('seed = ', seed)
noise = 0.1
#model = RandomContextualLinearArms(n_actions=4, n_features=2, noise=noise, random_state=seed, bound_context=1)
model = AttackOneUserModel(n_actions=25, n_features=20, noise=noise, random_state=seed, bound_context=1, distance=1/2)
model.add_target_arm()
model.thetas = model.thetas
theta_bound = np.max(np.linalg.norm(model.thetas, 2, axis=(1)))
target_context = np.random.randint(low=0, high=len(model.context_lists))
# other_context = np.random.randint(low=0, high=len(model.context_lists))
# while other_context == target_context:
#     other_context = np.random.randint(low=0, high=len(model.context_lists))
x_star = model.context_lists[target_context]
means_x_star = np.dot(model.thetas, x_star)
#target_arm = np.random.randint(low=0, high=model.n_actions)
# target_arm = np.argmin(means_x_star)
target_arm = model.n_actions-1
T = int(1*10**5)
nb_simu = 1
mask = np.ones(model.n_actions, dtype='int')
mask[target_arm] = 0
print('a star=', target_arm)
print('x_star', x_star)
print('means for context x_star:', np.dot(model.thetas, x_star))
print(in_hull(x=model.thetas[target_arm], points=np.array(model.thetas[mask])))
if in_hull(x=model.thetas[target_arm], points=np.array(model.thetas[mask])):
     raise ValueError()


algorithms = {
    'LinUCB': ContextualLinearBandit(nb_arms=model.n_actions, dimension=model.n_features,
                                     reg_factor=0.1, delta=0.01,
                                     bound_features=np.max(np.linalg.norm(model.thetas, axis=1)),
                                     noise_variance=noise, bound_context=model.bound_context),
    # 'LinUCB-Attacked': ContextualLinearBandit(nb_arms=model.n_actions, dimension=model.n_features,
    #                                  reg_factor=0.1, delta=0.99,
    #                                  bound_features=np.max(np.linalg.norm(model.thetas, axis=1)),
    #                                  noise_variance=noise, bound_context=model.bound_context),
    'LinUCB-RelaxedAttacked': ContextualLinearBandit(nb_arms=model.n_actions, dimension=model.n_features,
                                              reg_factor=0.1, delta=0.01,
                                              bound_features=np.max(np.linalg.norm(model.thetas, axis=1)),
                                              noise_variance=noise, bound_context=model.bound_context)
}
results = []
for alg_name, alg in algorithms.items():
    regret = np.zeros((nb_simu, T))
    nb_target_arms = np.zeros((nb_simu, T))
    nb_attack_needed = np.zeros((nb_simu, T))
    attack_condition = np.zeros((nb_simu, T))
    draws = [[]]*nb_simu
    context_norm = np.zeros((nb_simu, T))
    epsilon_norm = np.zeros((nb_simu, T))
    thetas_alg = np.zeros((nb_simu, T, model.n_actions, model.n_features))
    prod_scalar = np.zeros((nb_simu, T, model.n_actions, model.n_user))
    for k in trange(nb_simu, desc='Simulating {}'.format(alg_name)):
        alg.reset()
        for t in trange(T):
            context = model.get_context()
            old_context = context
            if 'Attacked' in alg_name:
                if np.linalg.norm(context - x_star) < 10**-10:
                    if 'Relaxed' in alg_name:
                        epsilon = compute_relaxed_attack(alg, target_arm, context, slack=10**-4)
                    else:
                        epsilon = compute_attack(alg, target_arm, context, slack=10 ** -3)
                else:
                    epsilon = np.zeros((model.n_features,))
                context = context + epsilon
                epsilon_norm[k, t] = np.linalg.norm(epsilon)
            a_t = alg.get_action(context)
            if np.linalg.norm(x_star - old_context) < 10**-10:
                draws[k].append(a_t)
            thetas_alg[k, t] = alg.thetas_hat
            # for a in range(model.n_actions):
            #     for i, x in enumerate(model.context_lists):
            #         p = np.dot(model.thetas[a], x) - np.dot(alg.thetas_hat[a], x)
            #         prod_scalar[k, t, a, i] = p
            r_t = model.reward(old_context, a_t)
            alg.update(context, a_t, r_t)
            regret[k, t] = model.best_arm_reward(old_context) - np.dot(model.thetas[a_t], old_context)
            context_norm[k, t] = np.linalg.norm(context)
        draws[k] = np.array(draws[k])
    draws = np.array(draws)
    print(draws.shape)
    print('Norm attacks=', epsilon_norm)
    results += [(alg_name, MABResults(regret=regret, attack_cond=attack_condition, target_draws=draws,
                                      thetas=thetas_alg, design_matrix=alg.inv_design_matrices, context_norm=context_norm, betas=alg.alpha()))]

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

    mean_condition = np.mean(np.cumsum(val.target_draws == target_arm, axis=1), axis=0)
    low_quantile = np.quantile(val.attack_cond, 0.1, axis=0)
    high_quantile = np.quantile(val.attack_cond, 0.9, axis=0)
    plt.figure(1)
    plt.title('Draws target arm')
    plt.plot(mean_condition, label=alg_name)
    # plt.fill_between(t, low_quantile, high_quantile, alpha=0.15)
    plt.legend()
    ### TODO: Change this plot to have the number of time the context has been presented in x axis

    if 'Attacked' in alg_name:
        plt.figure(2)
        plt.title('Cumulative attack norm attacked context')
        plt.plot(np.mean(np.cumsum(epsilon_norm, axis=1), axis=0), label=alg_name)
        low_quantile = np.quantile(np.cumsum(epsilon_norm, axis=1), 0.1, axis=0)
        high_quantile = np.quantile(np.cumsum(epsilon_norm, axis=1), 0.9, axis=0)
        plt.fill_between(t, low_quantile, high_quantile, alpha=0.15)
        plt.legend()

if model.n_features == 2:
    for i, (alg_name, val) in enumerate(results):
        plt.figure(i + 3)
        plt.title('Confidence ellipse for {}'.format(alg_name))
        x = np.linspace(0, 2*np.pi)
        x_1 = np.cos(x)
        y_1 = np.sin(x)
        X = np.vstack((x_1, y_1))
        betas = val.betas
        for a in range(model.n_actions):
            center = val.thetas[-1, -1, a]
            V = sqrtm(val.design_matrix[a])
            y = center.reshape((2, 1)) + betas[a] * np.dot(V, X)
            plt.plot(y[0, :], y[1, :], label = 'confidence ellipse arm {}'.format(a))
            plt.fill_between(y[0,:], y[1,:], (center.reshape((2, 1))*np.ones((2, 50)))[1, :], alpha=0.15)
            plt.scatter(model.thetas[a][0],model.thetas[a][1], c=new_colors[a])
            plt.scatter(center[0], center[1], marker='^', c=new_colors[a])
        plt.scatter(x_star[0], x_star[1], marker='+', c = new_colors[-1])
        plt.legend()
plt.show()
    #
    # plt.figure(4)
    # plt.title('Error true env and learned env attack context')
    # for a in range(model.n_actions):
    #     error = np.linalg.norm(val.thetas[:, :, a] - model.thetas[a], axis=2)
    #     plt.plot(t, np.mean(error, axis=0), label=alg_name + ' arm {}'.format(a))
    #     low_quantile = np.quantile(error, 0.1, axis=0)
    #     high_quantile = np.quantile(error, 0.9, axis=0)
    #     plt.fill_between(t, low_quantile, high_quantile, alpha=0.15)
    # plt.legend()


    # plt.figure(7)
    # plt.title('Error biased env and learned env attack context')
    # for a in range(model.n_actions):
    #     error = np.linalg.norm(val.thetas[:, :, a] - model.thetas[a]/np.maximum(2*np.dot(model.thetas[a], x_star)/np.dot(model.thetas[target_arm], x_star), 1), axis=2)
    #     plt.plot(t, np.mean(error, axis=0), label=alg_name + ' arm {}'.format(a))
    #     low_quantile = np.quantile(error, 0.1, axis=0)
    #     high_quantile = np.quantile(error, 0.9, axis=0)
    #     plt.fill_between(t, low_quantile, high_quantile, alpha=0.15)
    # plt.legend()

    # plt.figure(5)
    # plt.title('Difference estimated reward for target context {}'.format(target_context))
    # for a in range(model.n_actions):
    #     plt.plot(t, np.mean(val.prod_scalar[:, :, a, target_context], axis=0), label=alg_name + ' arm {}'.format(a))
    #     low_quantile = np.quantile(val.prod_scalar[:, :, a, target_context], 0.1, axis=0)
    #     high_quantile = np.quantile(val.prod_scalar[:, :, a, target_context], 0.9, axis=0)
    #     plt.fill_between(t, low_quantile, high_quantile, alpha=0.15)
    # plt.legend()


    # plt.figure(6)
    # plt.title('Difference estimated reward for a random non target context {}'.format(other_context))
    # for a in range(model.n_actions):
    #     plt.plot(t, np.mean(val.prod_scalar[:, :, a, other_context], axis=0), label=alg_name + ' arm {}'.format(a))
    #     low_quantile = np.quantile(val.prod_scalar[:, :, a, other_context], 0.1, axis=0)
    #     high_quantile = np.quantile(val.prod_scalar[:, :, a, other_context], 0.9, axis=0)
    #     plt.fill_between(t, low_quantile, high_quantile, alpha=0.15)
    # plt.legend()


plt.show()

#print('Target arms=', np.mean(np.cumsum(nb_target_arms,axis=1)))
#print('Attack needed arms=', np.mean(np.cumsum(nb_attack_needed,axis=1)))
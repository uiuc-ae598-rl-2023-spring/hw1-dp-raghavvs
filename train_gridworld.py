import numpy as np
import matplotlib.pyplot as plt
import gridworld
import algorithms
import plots
import time

####### Model Initialization ########

env = gridworld.GridWorld(hard_version=False)
s = env.reset()
gamma = 0.95
theta = 1e-8

######## Policy Iteration ########

agent = algorithms.PolicyIteration(env, gamma)

# Train the agent
start_time = time.time()
V_list, policy, total_iterations, _, vmv, policy_star = agent.train()
#end_time = time.time()
#print(f'Policy iteration for Gridworld problem - Total time taken: {end_time - start_time:.2f} seconds')

# Plot 1 - Mean value function versus number of iterations
mean_value = np.mean(np.array(V_list), axis=1)
iterations = np.arange(len(V_list))

plt.figure(figsize = (8, 6))
plt.plot(range(25), vmv)
plt.xlabel('Iterations')
plt.ylabel('Mean value function')
plt.title('Policy Iteration - Gridworld - Value function')
plt.savefig('figures/gridworld/g_pi_valfn_vs_iter.png')
#plt.show()

# Plot 2 - Policy and trajectory
plt.figure(figsize = (8, 6))
log = plots.trajectory(env, policy_star)
plt.plot(log['t'], log['s'], '-o')
plt.plot(log['t'][:-1], log['a'])
plt.plot(log['t'][:-1], log['r'])
plt.title('Policy Iteration - Gridworld - Trajectory')
plt.legend(['s', 'a', 'r'])
plt.savefig('figures/gridworld/g_pi_policy_vs_trajec.png')
#plt.show

######## Value Iteration ########

agent = algorithms.ValueIteration(env, gamma, theta)

# Train the agent
#start_time = time.time()
_, policy = agent.value_iteration()
#end_time = time.time()
#print(f'Value iteration for Gridworld problem - Total time taken: {end_time - start_time:.2f} seconds')

# Plot 1 - Mean value function versus number of iterations
num_iterations = 50
mean_values = []

for i in range(num_iterations):
    agent.value_iteration()
    mean_value = np.mean(agent.values)
    mean_values.append(mean_value)

#print(agent.values.reshape(5,5))

plt.figure(figsize = (8, 6))
plt.plot(range(num_iterations), mean_values)
plt.xlabel('Number of Iterations')
plt.ylabel('Mean Value Function')
plt.title('Value Iteration Method')
plt.savefig('figures/gridworld/g_vi_valfn_vs_iter.png')
#plt.show()

# Plot 2 - Policy and trajectory
plt.figure(figsize = (8, 6))
log = plots.trajectory(env, policy)
plt.plot(log['t'], log['s'], '-o')
plt.plot(log['t'][:-1], log['a'])
plt.plot(log['t'][:-1], log['r'])
plt.title('Policy Iteration - Gridworld - Trajectory')
plt.legend(['s', 'a', 'r'])
plt.savefig('figures/gridworld/g_vi_policy_vs_trajec.png')
#plt.show

####### Model Initialization ########

num_episodes = 500
gamma = 0.95
epsilons = [0, 0.2, 0.4, 0.6, 0.8, 1]
alphas = [0.2, 0.4, 0.5, 0.6, 0.8, 1]
alpha = 0.1

######## SARSA ########

# Train the agent
Q, Q_list, _ = algorithms.sarsa(env, alpha=0.1, epsilon=0.1, gamma=0.95, num_episodes=500)

policy= np.argmax(Q, axis = 1)

# TD(0) evaluation
V = algorithms.TD0(env, policy, alpha=0.1, gamma=0.95, total_episodes=500)
title = r"$\alpha = {}, \gamma = {}$".format(alpha, gamma)

# Plot 1 - SARSA Learning curve plot
plots.plot_v_episodes(Q_list, "SARSA", "Number of episodes", "Mean Q", "Gridworld, " + title, 'r')
plt.savefig("figures/gridworld/g_sarsa_learning_curve.png")

# Plot 2 - SARSA State value function plot
plots.plot_v_episodes(Q, "SARSA", "State", r"$V(s)$", r"Gridworld, $V^{*}(s)$, TD(0), " + title, 'r')
plt.savefig("figures/gridworld/g_sarsa_state_value.png")

# Plot 3 - SARSA Policy plot
plots.plot_v_episodes(policy, "SARSA", "State", "$\pi(s)$", r"Gridworld, $\pi^{*}$, " + title, 'r')
plt.savefig("figures/gridworld/g_sarsa_policy.png")

# Plot 4 - SARSA Learning Curves for Different Alpha Values
v_name   = r"$\alpha = $"
plt.figure(figsize = (8, 6))
for alpha in alphas:    
    _, Q_list, _ = algorithms.sarsa(env, alpha, epsilon=0.1, gamma=0.95, num_episodes=500)
    plots.plot_q_episodes(Q_list, alpha, v_name)
plt.xlabel("Number of episodes")
plt.ylabel("Mean Value Function")
plt.title("SARSA -  Gridworld, "  + r"$\epsilon = 0.1$, " + "$\gamma = {}$".format(gamma))
plt.legend()
plt.savefig("figures/gridworld/g_sarsa_learning_curves_alpha2.png")
#plt.show()

# Plot 5 - SARSA Learning Curves for Different Epsilon Values
v_name   = r"$\epsilon = $"
plt.figure(figsize=(8, 6))
for epsilon in epsilons:    
    _, Q_list, _ = algorithms.sarsa(env, alpha, epsilon, gamma=0.95, num_episodes=500)
    plots.plot_q_episodes(Q_list, epsilon, v_name)
epsilon = np.arange(len(Q_list)) + 1
plt.plot(epsilon, Q_list)
plt.xlabel("Number of episodes")
plt.ylabel("Mean Value Function")
plt.title("SARSA, Gridworld, "  + r"$\alpha = {}$, $\gamma = {}$".format(alpha, gamma))
plt.legend()
plt.savefig("figures/gridworld/g_sarsa_learning_curves_epsilon2.png")
#plt.show()

# Plot 6 - SARSA Policy and trajectory
plt.figure(figsize = (8, 6))
log = plots.trajectory(env, policy)
plt.plot(log['t'], log['s'], '-o')
plt.plot(log['t'][:-1], log['a'])
plt.plot(log['t'][:-1], log['r'])
plt.title('SARSA - Gridworld - Trajectory')
plt.legend(['s', 'a', 'r'])
plt.savefig('figures/gridworld/g_sarsa_policy_vs_trajec.png')
#plt.show

######## Q-Learning ########

# Train the agent
#start_time = time.time()
Q, Q_list, _, policy, _ = algorithms.q_learning(env, alpha=0.1, epsilon=0.1, gamma=0.95, num_episodes=500)
#end_time = time.time()
#print(f'Q-Learning for Gridworld problem - Total time taken: {end_time - start_time:.2f} seconds')

# Plot 1 - Q-Learning Learning curve plot
plots.plot_v_episodes(Q_list, "Q-Learning", "Number of episodes", "Mean Q", "Gridworld, " + title, 'r')
plt.savefig("figures/gridworld/g_ql_learning_curve.png")

# Plot 2 - Q-Learning State value function plot
plots.plot_v_episodes(Q, "Q-Learning", "State", r"$V(s)$", r"Gridworld, $V^{*}(s)$, TD(0), " + title, 'r')
plt.savefig("figures/gridworld/g_ql_state_value.png")

# Plot 3 - Q-Learning Policy plot
plots.plot_v_episodes(policy, "Q-Learning", "State", "$\pi(s)$", r"Gridworld, $\pi^{*}$, " + title, 'r')
plt.savefig("figures/gridworld/g_ql_policy.png")

# Plot 4 - Q-Learning Learning Curves for Different Alpha Values
v_name   = r"$\alpha = $"
plt.figure(figsize = (8, 6))
for alpha in alphas:    
    _, Q_list, _, _, _ = algorithms.q_learning(env, alpha, epsilon=0.1, gamma=0.95, num_episodes=500)
    plots.plot_q_episodes(Q_list, alpha, v_name)
plt.xlabel("Number of episodes")
plt.ylabel("Mean Value Function")
plt.title("Q-Learning -  Gridworld, "  + r"$\epsilon = 0.1$, " + "$\gamma = {}$".format(gamma))
plt.legend()
plt.savefig("figures/gridworld/g_ql_learning_curves_alpha2.png")
#plt.show()

# Plot 5 - Q-Learning Learning Curves for Different Epsilon Values
v_name   = r"$\epsilon = $"
plt.figure(figsize=(8, 6))
for epsilon in epsilons:    
    _, Q_list, _, _, _ = algorithms.q_learning(env, alpha, epsilon, gamma=0.95, num_episodes=500)
    plots.plot_q_episodes(Q_list, epsilon, v_name)
epsilon = np.arange(len(Q_list)) + 1
plt.plot(epsilon, Q_list)
plt.xlabel("Number of episodes")
plt.ylabel("Mean Value Function")
plt.title("Q-Learning, Gridworld, "  + r"$\alpha = {}$, $\gamma = {}$".format(alpha, gamma))
plt.legend()
plt.savefig("figures/gridworld/g_ql_learning_curves_epsilon2.png")
#plt.show()

# Plot 6 - Q-Learning Policy and trajectory
plt.figure(figsize = (8, 6))
log = plots.trajectory(env, policy)
plt.plot(log['t'], log['s'], '-o')
plt.plot(log['t'][:-1], log['a'])
plt.plot(log['t'][:-1], log['r'])
plt.title('Q-Learning - Gridworld - Trajectory')
plt.legend(['s', 'a', 'r'])
plt.savefig('figures/gridworld/g_ql_policy_vs_trajec.png')
#plt.show

end_time = time.time()
print(f'Gridworld problem - Total time taken: {end_time - start_time:.2f} seconds')
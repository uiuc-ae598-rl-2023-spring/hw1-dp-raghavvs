import numpy as np
import matplotlib.pyplot as plt
import discrete_pendulum
import algorithms
import plots
import time

####### Model Initialization ########

env = discrete_pendulum.Pendulum(n_theta=15, n_thetadot=21)
s = env.reset()
gamma = 0.95
theta = 1e-8
num_episodes = 500
epsilons = [0, 0.2, 0.4, 0.6, 0.8, 1]
alphas = [0.2, 0.4, 0.5, 0.6, 0.8, 1]
alpha = 0.1
epsilon = 1

######## SARSA ########

# Train the agent
start_time = time.time()
Q, Q_list, _ = algorithms.sarsa(env, alpha=0.1, epsilon=0.1, gamma=0.95, num_episodes=500)

policy= np.argmax(Q, axis = 1)

# TD(0) evaluation
V = algorithms.TD0(env, policy, alpha=0.1, gamma=0.95, total_episodes=500)
title = r"$\alpha = {}, \gamma = {}$".format(alpha, gamma)

# Plot 1 - SARSA Learning curve plot
plots.plot_v_episodes(Q_list, "SARSA", "Number of episodes", "Mean Q", "Pendulum, " + title, 'r')
plt.savefig("figures/pendulum/p_sarsa_learning_curve.png")

# Plot 2 - SARSA State value function plot
plots.plot_v_episodes(Q, "SARSA", "State", r"$V(s)$", r"Pendulum, $V^{*}(s)$, TD(0), " + title, 'r')
plt.savefig("figures/pendulum/p_sarsa_state_value.png")

# Plot 3 - SARSA Policy plot
plots.plot_v_episodes(policy, "SARSA", "State", "$\pi(s)$", r"Pendulum, $\pi^{*}$, " + title, 'r')
plt.savefig("figures/pendulum/p_sarsa_policy.png")

# Plot 4 - SARSA Learning Curves for Different Alpha Values
v_name   = r"$\alpha = $"
plt.figure(figsize = (8, 6))
for alpha in alphas:    
    _, Q_list, _ = algorithms.sarsa(env, alpha, epsilon=0.1, gamma=0.95, num_episodes=500)
    plots.plot_q_episodes(Q_list, alpha, v_name)
plt.xlabel("Number of episodes")
plt.ylabel("Mean Value Function")
plt.title("SARSA -  Pendulum, "  + r"$\epsilon = 0.1$, " + "$\gamma = {}$".format(gamma))
plt.legend()
plt.savefig("figures/pendulum/p_sarsa_learning_curves_alpha.png")
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
plt.title("SARSA, Pendulum, "  + r"$\alpha = {}$, $\gamma = {}$".format(alpha, gamma))
plt.legend()
plt.savefig("figures/pendulum/p_sarsa_learning_curves_epsilon.png")
#plt.show()

# Plot 6 - SARSA Policy and trajectory
log = plots.trajectory_p(env, policy)
fig, ax = plt.subplots(2, 1, figsize=(10, 10))
ax[0].plot(log['t'], log['s'])
ax[0].plot(log['t'][:-1], log['a'])
ax[0].plot(log['t'][:-1], log['r'])
ax[0].legend(['s', 'a', 'r'])
ax[1].plot(log['t'], log['theta'])
ax[1].plot(log['t'], log['thetadot'])
ax[1].legend(['theta', 'thetadot'])
plt.savefig('figures/pendulum/p_sarsa_policy_vs_trajec_new.png') 

######## Q-Learning ########

# Train the agent
#start_time = time.time()
Q, Q_list, _, policy, _ = algorithms.q_learning(env, alpha=0.1, epsilon=0.1, gamma=0.95, num_episodes=500)
#end_time = time.time()
#print(f'Q-Learning for Pendulum problem - Total time taken: {end_time - start_time:.2f} seconds')

# Plot 1 - Q-Learning Learning curve plot
plots.plot_v_episodes(Q_list, "Q-Learning", "Number of episodes", "Mean Q", "Pendulum, " + title, 'r')
plt.savefig("figures/pendulum/p_ql_learning_curve.png")

# Plot 2 - Q-Learning State value function plot
plots.plot_v_episodes(Q, "Q-Learning", "State", r"$V(s)$", r"Pendulum, $V^{*}(s)$, TD(0), " + title, 'r')
plt.savefig("figures/pendulum/p_ql_state_value.png")

# Plot 3 - Q-Learning Policy plot
plots.plot_v_episodes(policy, "Q-Learning", "State", "$\pi(s)$", r"Pendulum, $\pi^{*}$, " + title, 'r')
plt.savefig("figures/pendulum/p_ql_policy.png")

# Plot 4 - Q-Learning Learning Curves for Different Alpha Values
v_name   = r"$\alpha = $"
plt.figure(figsize = (8, 6))
for alpha in alphas:    
    _, Q_list, _, _, _ = algorithms.q_learning(env, alpha, epsilon=0.1, gamma=0.95, num_episodes=500)
    plots.plot_q_episodes(Q_list, alpha, v_name)
plt.xlabel("Number of episodes")
plt.ylabel("Mean Value Function")
plt.title("Q-Learning -  Pendulum, "  + r"$\epsilon = 0.1$, " + "$\gamma = {}$".format(gamma))
plt.legend()
plt.savefig("figures/pendulum/p_ql_learning_curves_alpha.png")
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
plt.title("Q-Learning, Pendulum, "  + r"$\alpha = {}$, $\gamma = {}$".format(alpha, gamma))
plt.legend()
plt.savefig("figures/pendulum/p_ql_learning_curves_epsilon.png")
#plt.show()

# Plot 6 - Q-Learning Policy and trajectory
log = plots.trajectory_p(env, policy)
fig, ax = plt.subplots(2, 1, figsize=(10, 10))
ax[0].plot(log['t'], log['s'])
ax[0].plot(log['t'][:-1], log['a'])
ax[0].plot(log['t'][:-1], log['r'])
ax[0].legend(['s', 'a', 'r'])
ax[1].plot(log['t'], log['theta'])
ax[1].plot(log['t'], log['thetadot'])
ax[1].legend(['theta', 'thetadot'])
plt.savefig('figures/pendulum/p_ql_policy_vs_trajec_new.png')

end_time = time.time()
print(f'Pendulum problem - Total time taken: {end_time - start_time:.2f} seconds')

# Pendulum problem - Total time taken: 1537.41 seconds
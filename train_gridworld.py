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
end_time = time.time()
print(f'Policy iteration for Gridworld problem - Total time taken: {end_time - start_time:.2f} seconds')

# Plot 1 - Mean value function versus number of iterations
mean_value = np.mean(np.array(V_list), axis=1)
iterations = np.arange(len(V_list))

plt.figure(figsize = (8, 6))
plt.plot(range(25), vmv)
plt.xlabel('Iterations')
plt.ylabel('Mean value function')
plt.title('Policy Iteration - Gridworld - Value function')
plt.grid()
plt.savefig('figures/gridworld/pi_valfn_vs_iter.png')
#plt.show()

# Plot 2 - Policy and trajectory
plt.figure(figsize = (8, 6))
log = plots.trajectory(env, policy_star)
plt.plot(log['t'], log['s'], '-o')
plt.plot(log['t'][:-1], log['a'])
plt.plot(log['t'][:-1], log['r'])
plt.title('Policy Iteration - Gridworld - Trajectory')
plt.grid()
plt.legend(['s', 'a', 'r'])
plt.savefig('figures/gridworld/pi_policy_vs_trajec.png')
#plt.show

######## Value Iteration ########

agent = algorithms.ValueIteration(env, gamma, theta)

# Train the agent
start_time = time.time()
_, policy = agent.value_iteration()
end_time = time.time()
print(f'Value iteration for Gridworld problem - Total time taken: {end_time - start_time:.2f} seconds')

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
plt.savefig('figures/gridworld/vi_valfn_vs_iter.png')
#plt.show()

# Plot 2 - Policy and trajectory
plt.figure(figsize = (8, 6))
log = plots.trajectory(env, policy)
plt.plot(log['t'], log['s'], '-o')
plt.plot(log['t'][:-1], log['a'])
plt.plot(log['t'][:-1], log['r'])
plt.title('Policy Iteration - Gridworld - Trajectory')
plt.grid()
plt.legend(['s', 'a', 'r'])
plt.savefig('figures/gridworld/vi_policy_vs_trajec.png')
#plt.show

######## SARSA ########

# Plot 1 - Return vs. Episodes
num_episodes = 500
# Train the agent
start_time = time.time()
_, returns = algorithms.sarsa(env, num_episodes=num_episodes)
end_time = time.time()
print(f'SARSA for Gridworld problem - Total time taken: {end_time - start_time:.2f} seconds')

plt.figure(figsize = (8, 6))
plt.plot(range(num_episodes), returns)
plt.xlabel('Episode')
plt.ylabel('Return')
plt.title('SARSA Learning Curve')
plt.savefig('figures/gridworld/sarsa_return_vs_episodes.png')
plt.show()

# Plot 2 - SARSA Learning Curves for Different Epsilon Values
epsilons = [0.1, 0.2, 0.3, 0.4, 0.5]

plt.figure(figsize = (8, 6))
fig, ax = plt.subplots()
for epsilon in epsilons:
    _, returns = algorithms.sarsa(env, epsilon=epsilon, num_episodes=num_episodes)
    ax.plot(range(num_episodes), returns, label=f'epsilon={epsilon}')
ax.legend()
ax.set_xlabel('Episode')
ax.set_ylabel('Return')
ax.set_title('SARSA Learning Curves for Different Epsilon Values')
plt.savefig('figures/gridworld/sarsa_learning_curves_epsilon.png')
plt.show()

# Plot 3 - SARSA Learning Curves for Different Alpha Values 
alphas = [0.1, 0.2, 0.3, 0.4, 0.5]

plt.figure(figsize = (8, 6))
fig, ax = plt.subplots()
for alpha in alphas:
    _, returns = algorithms.sarsa(env, alpha=alpha, num_episodes=num_episodes)
    ax.plot(range(num_episodes), returns, label=f'alpha={alpha}')
ax.legend()
ax.set_xlabel('Episode')
ax.set_ylabel('Return')
ax.set_title('SARSA Learning Curves for Different Alpha Values')
plt.savefig('figures/gridworld/sarsa_learning_curves_alpha.png')
plt.show()
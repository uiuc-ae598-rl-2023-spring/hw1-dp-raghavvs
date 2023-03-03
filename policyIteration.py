import numpy as np
import matplotlib.pyplot as plt
import gridworld
import time

class PolicyIteration:
    def __init__(self, env, gamma, theta):
        self.env = env
        self.gamma = gamma
        self.theta = 1e-8
        self.policy = np.ones([env.num_states, env.num_actions]) / env.num_actions

    def policy_evaluation(self, V, policy):
        num_iterations = 0
        vmv = []
        while True:
            delta = 0
            delta_list = []
            for s in range(self.env.num_states):
                v = 0
                for a, action_prob in enumerate(policy[s]):
                    for next_s in range(self.env.num_states):
                        reward = self.env.r(s, a)
                        prob = self.env.p(next_s, s, a)
                        v += action_prob * prob * (reward + self.gamma * V[next_s])
                delta = max(delta, np.abs(v - V[s]))
                V[s] = v
                V_mean = np.mean(V[s])
                vmv.append(V_mean)
            num_iterations += 1
            delta_list.append(delta)
            if delta < self.theta:
                break
        return num_iterations, delta_list, vmv

    def policy_improvement(self, V, policy):
        policy_stable = True
        for s in range(self.env.num_states):
            old_action = np.argmax(policy[s])
            action_values = np.zeros(self.env.num_actions)
            for a in range(self.env.num_actions):
                for next_s in range(self.env.num_states):
                    reward = self.env.r(s, a)
                    prob = self.env.p(next_s, s, a)
                    action_values[a] += prob * (reward + self.gamma * V[next_s])
            best_action = np.argmax(action_values)
            policy[s] = np.zeros(self.env.num_actions)
            policy[s][best_action] = 1.0
            if old_action != best_action:
                policy_stable = False
        return policy_stable

    def train(self):
        V = np.zeros(self.env.num_states)
        V_list = []
        num_iterations_list = []
        delta_list = [] 
        while True:
            num_iterations, delta, vmv = self.policy_evaluation(V, self.policy)
            num_iterations_list.append(num_iterations)
            delta_list.extend(delta) 
            policy_stable = self.policy_improvement(V, self.policy)
            V_list.append(V.copy())
            if policy_stable:
                total_iterations = sum(num_iterations_list)
                return V_list, self.policy, total_iterations, delta_list, vmv

env = gridworld.GridWorld()
agent = PolicyIteration(env)

# Train the agent
start_time = time.time()
V_list, policy, total_iterations, _, vmv = agent.train()
end_time = time.time()
print(f'Total time taken: {end_time - start_time:.4f} seconds')

# Plot the mean value function versus number of iterations
mean_value = np.mean(np.array(V_list), axis=1)
iterations = np.arange(len(V_list))

#print(vmv)

plt.plot(range(25), vmv)
plt.xlabel('Iterations')
plt.ylabel('Mean value')
plt.title('Mean value function versus number of iterations')
plt.savefig('figures/gridworld/pi_valfn_vs_iter.png')
#plt.show()

""" env = gridworld.GridWorld()

pi = PolicyIteration(env)

start_time = time.time()
V_list, policy, total_iterations, delta_list = pi.train()
end_time = time.time()
training_time = end_time - start_time

print(V_list)
#print(delta_list)

# Plot 
fig, ax = plt.subplots()
for i, V in enumerate(V_list):
    ax.plot(np.arange(env.num_states), V, label=f"iteration {i+1}")
ax.set_xlabel("State")
ax.set_ylabel("Value Function")
ax.legend()
plt.savefig('figures/gridworld/pi_valfn_vs_iter.png')
#plt.show()

print(f'Total training time: {training_time:.2f} seconds')
print(f'Total number of iterations: {total_iterations}') """

"""
Output:
V = 
[41.99469264 44.20493963 41.99469264 39.20493963 37.24469264 39.89495801
 41.99469264 39.89495801 37.90021011 36.00519961 37.90021011 39.89495801
 37.90021011 36.00519961 34.20493963 36.00519961 37.90021011 36.00519961
 34.20493963 32.49469264 34.20493963 36.00519961 34.20493963 32.49469264
 30.86995801]

delta_list = 
[9.20708931317904e-09, 8.678249230342772e-09, 1.4210854715202004e-14, 
 7.105427357601002e-15, 7.105427357601002e-15]

----
Need to verify plot.
"""
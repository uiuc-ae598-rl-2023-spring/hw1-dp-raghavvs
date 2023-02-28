import numpy as np
import matplotlib.pyplot as plt
import gridworld
import time

class PolicyIteration:
    def __init__(self, env, gamma=0.95):
        self.env = env
        self.gamma = gamma
        self.policy = np.ones([env.num_states, env.num_actions]) / env.num_actions

    def policy_evaluation(self, V, policy):
        theta = 1e-8
        num_iterations = 0
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
            num_iterations += 1
            delta_list.append(delta)
            if delta < theta:
                break
        return num_iterations, delta_list

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
        num_iterations_list = []
        delta_list = [] 
        while True:
            num_iterations, delta = self.policy_evaluation(V, self.policy)
            num_iterations_list.append(num_iterations)
            delta_list.extend(delta) 
            policy_stable = self.policy_improvement(V, self.policy)
            if policy_stable:
                total_iterations = sum(num_iterations_list)
                return V, self.policy, total_iterations, delta_list


# Plot 1 - Value function vs. Iterations

env = gridworld.GridWorld()
policy_iteration = PolicyIteration(env)

iterations = []
mean_V = []

for i in range(25):
    V, policy, total_iterations, delta_list = policy_iteration.train()
    iterations.append(total_iterations)
    mean_V.append(np.mean(V))

plt.plot(iterations, mean_V)
plt.xlabel('Number of iterations')
plt.ylabel('Mean value function over 25 states')
#plt.ylim(0, 40)
time.sleep(40)
plt.savefig('figures/gridworld/pi_iter_vs_valfn.png')
#plt.show()

# Plot 2 - Error vs. Iterations

pi = PolicyIteration(env)

V, policy, total_iterations, delta_list = pi.train()

print(delta_list)

plt.plot(delta_list)
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.title('Error vs. Number of Iterations')
time.sleep(40)
plt.savefig('figures/gridworld/pi_iter_vs_error.png')
#plt.show()

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
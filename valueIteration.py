import numpy as np
import matplotlib.pyplot as plt
import gridworld
import time

class ValueIteration():
    def __init__(self, env, gamma, theta):
        self.env = env
        self.gamma = gamma
        self.theta = theta

        self.values = np.zeros(env.num_states)

    def value_iteration(self):
        delta = np.inf
        while delta > self.theta:
            delta = 0
            delta_diff = 0
            delta_list = []
            for s in range(self.env.num_states):
                v = self.values[s]
                max_action_value = -np.inf
                for a in range(self.env.num_actions):
                    action_value = 0
                    for s1 in range(self.env.num_states):
                        action_value += self.env.p(s1, s, a) * (self.env.r(s, a) + self.gamma * self.values[s1])
                    max_action_value = max(max_action_value, action_value)
                self.values[s] = max_action_value
                delta_diff = abs(v - self.values[s])
                #print(delta_diff)
                delta = max(delta, abs(v - self.values[s]))
            delta_list.append(delta_diff)
        return delta_list

# Plot 1 - Value function vs. Iterations

env = gridworld.GridWorld()

vi = ValueIteration(env, gamma=0.95, theta=1e-8)
vi.value_iteration()
#print(vi.values.reshape(5,5))

env_vi = ValueIteration(env, gamma=0.95, theta=1e-8)

num_iterations = 50
mean_values = []

for i in range(num_iterations):
    env_vi.value_iteration()
    mean_value = np.mean(env_vi.values)
    mean_values.append(mean_value)

#print(range(num_iterations))
#print(mean_values)

print(len(mean_values))

plt.plot(range(num_iterations), mean_values)
plt.xlabel('Number of Iterations')
plt.ylabel('Mean Value Function')
plt.title('Value Iteration Method')
#plt.ylim(0, 40)
time.sleep(40)
plt.savefig('figures/gridworld/vi_valfn_vs_iter.png')
#plt.show()

""" # Plot 2 - Error vs. Iterations

vi = ValueIteration(env, gamma=0.95, theta=1e-4)

delta_list = vi.value_iteration()

print(delta_list)

plt.plot(num_iterations, delta_list)
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.title('Error vs. Number of Iterations')
time.sleep(40)
plt.savefig('figures/gridworld/vi_iter_vs_error.png')
#plt.show() """

"""
Output:
[[41.99437669 44.20468228 41.99444817 39.20468228 37.24444817]
 [39.89465786 41.99444817 39.89472576 37.89998947 36.00499   ]
 [37.89992496 39.89472576 37.89998947 36.00499    34.2047405 ]
 [36.00492872 37.89998947 36.00499    34.2047405  32.49450347]
 [34.20468228 36.00499    34.2047405  32.49450347 30.8697783 ]]

------
 Need to verify plot.
"""
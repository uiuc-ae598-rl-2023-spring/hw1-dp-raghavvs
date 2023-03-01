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
        V_list = []
        num_iterations_list = []
        delta_list = [] 
        while True:
            num_iterations, delta = self.policy_evaluation(V, self.policy)
            num_iterations_list.append(num_iterations)
            delta_list.extend(delta) 
            policy_stable = self.policy_improvement(V, self.policy)
            V_list.append(V.copy())
            if policy_stable:
                total_iterations = sum(num_iterations_list)
                return V_list, self.policy, total_iterations, delta_list

env = gridworld.GridWorld()

pi = PolicyIteration(env)

start_time = time.time()
V_list, policy, total_iterations, delta_list = pi.train()
end_time = time.time()
training_time = end_time - start_time

fig, ax = plt.subplots()
for i, V in enumerate(V_list):
    ax.plot(np.arange(env.num_states), V, label=f"iteration {i+1}")
ax.set_xlabel("State")
ax.set_ylabel("Value Function")
ax.legend()
plt.savefig('figures/gridworld/pi_valfn_vs_iter.png')
plt.show()

print(f'Total training time: {training_time:.2f} seconds')
print(f'Total number of iterations: {total_iterations}')
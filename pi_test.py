import numpy as np
import matplotlib.pyplot as plt
import gridworld

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
            if delta < theta:
                break
        return num_iterations

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
        while True:
            num_iterations = self.policy_evaluation(V, self.policy)
            num_iterations_list.append(num_iterations)
            policy_stable = self.policy_improvement(V, self.policy)
            V_list.append(V.copy())
            if policy_stable:
                break
        total_iterations = sum(num_iterations_list)
        return V_list, self.policy, total_iterations

env = gridworld.GridWorld()
agent = PolicyIteration(env)
V_list, policy, total_iterations = agent.train()

# Plot the mean value function versus number of iterations
mean_value = np.mean(np.array(V_list[:total_iterations]), axis=1)
iterations = np.arange(total_iterations)
plt.plot(iterations, mean_value)
plt.xlabel('Iterations')
plt.ylabel('Mean value')
plt.title('Mean value function versus number of iterations')
plt.show()

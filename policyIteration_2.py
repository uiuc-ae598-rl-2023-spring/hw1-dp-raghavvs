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

    def plot_trajectory(self, start_state):
        state = start_state
        states = [state]
        rewards = [self.env.r(state, np.argmax(self.policy[state]))]
        while not self.env.is_terminal(state):
            state, reward = self.env.step(state, np.argmax(self.policy[state]))
            states.append(state)
            rewards.append(reward)
        plt.plot(states)
        plt.title(f'Trajectory from state {start_state}')
        plt.xlabel('Time steps')
        plt.ylabel('States')
        plt.show()

    def plot_policy(self):
        plt.figure(figsize=(5,5))
        ax = plt.gca()
        ax.set_xticks(np.arange(5))
        ax.set_yticks(np.arange(5))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(color='k', linestyle='-', linewidth=2)
        for i in range(5):
            for j in range(5):
                s = i * 5 + j
                a = np.argmax(self.policy[s])
                if self.env.is_terminal(s):
                    continue
                if a == 0:
                    plt.arrow(j + 0.5, i + 0.1, 0, 0.8, head_width=0.1, head_length=0.1, fc='k', ec='k')
                elif a == 1:
                    plt.arrow(j + 0.5, i + 0.9, 0, -0.8, head_width=0.1, head_length=0.1, fc='k', ec='k')
                elif a == 2:
                    plt.arrow(j + 0.1, i + 0.5, 0.8, 0, head_width=0.1, head_length=0.1, fc='k', ec='k')
                elif a == 3:
                    plt.arrow(j + 0.9, i + 0.5, -0.8, 0, head_width=0.1, head_length=0.1, fc='k', ec='k')
        plt.show()

env = gridworld.GridWorld()

agent = PolicyIteration(env)
V, policy, total_iterations, delta_list = agent.train()
agent.plot_policy()


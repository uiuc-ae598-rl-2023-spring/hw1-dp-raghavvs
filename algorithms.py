import random
import numpy as np

######## Policy Iteration ########

class PolicyIteration:
    def __init__(self, env, gamma):
        self.env = env
        self.gamma = gamma
        self.policy = np.ones([env.num_states, env.num_actions]) / env.num_actions

    def policy_evaluation(self, V, policy):
        theta = 1e-8
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
            if delta < theta:
                break
        return num_iterations, delta_list, vmv

    def policy_improvement(self, V, policy):
        policy_stable = True
        policy_star = []
        for s in range(self.env.num_states):
            old_action = np.argmax(policy[s])
            action_values = np.zeros(self.env.num_actions)
            for a in range(self.env.num_actions):
                for next_s in range(self.env.num_states):
                    reward = self.env.r(s, a)
                    prob = self.env.p(next_s, s, a)
                    action_values[a] += prob * (reward + self.gamma * V[next_s])
            best_action = np.argmax(action_values)
            policy_star.append(np.argmax(action_values))
            policy[s] = np.zeros(self.env.num_actions)
            policy[s][best_action] = 1.0
            if old_action != best_action:
                policy_stable = False
        return policy_stable, policy_star

    def train(self):
        V = np.zeros(self.env.num_states)
        V_list = []
        num_iterations_list = []
        delta_list = [] 
        while True:
            num_iterations, delta, vmv = self.policy_evaluation(V, self.policy)
            num_iterations_list.append(num_iterations)
            delta_list.extend(delta) 
            policy_stable, policy_star = self.policy_improvement(V, self.policy)
            V_list.append(V.copy())
            if policy_stable:
                total_iterations = sum(num_iterations_list)
                return V_list, self.policy, total_iterations, delta_list, vmv, policy_star

######## Value Iteration ########

class ValueIteration():
    def __init__(self, env, gamma, theta):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.values = np.zeros(env.num_states)

    def value_iteration(self):
        while True:
            delta = 0
            for s in range(self.env.num_states):
                v = self.values[s]
                max_action_value = -np.inf
                for a in range(self.env.num_actions):
                    action_value = 0
                    for s1 in range(self.env.num_states):
                        action_value += self.env.p(s1, s, a) * (self.env.r(s, a) + self.gamma * self.values[s1])
                    max_action_value = max(max_action_value, action_value)
                self.values[s] = max_action_value
                delta = max(delta, abs(v - self.values[s]))
            if delta < self.theta:
                break
        policy = np.zeros(self.env.num_states, dtype=int)
        for s in range(self.env.num_states):
            max_action_value = -np.inf
            for a in range(self.env.num_actions):
                action_value = 0
                for s1 in range(self.env.num_states):
                    action_value += self.env.p(s1, s, a) * (self.env.r(s, a) + self.gamma * self.values[s1])
                if action_value > max_action_value:
                    max_action_value = action_value
                    policy[s] = a
        return self.values, policy

######## SARSA TD(0) ########

def epsilon_greedy_policy(env, Q, state, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        return np.random.choice(range(env.num_actions))
    else:
        return np.argmax(Q[state, :])
    
def TD0(env, policy, alpha, gamma, total_episodes):
    V = np.zeros(env.num_states)
    for eps in range(total_episodes):
        s = env.reset()
        t = 0
        while t < env.max_num_steps:
            a = policy[s]
            s1, reward, done = env.step(a)
            target  = reward + gamma*V[s1]
            V[s] += alpha * (target - V[s])
            s = s1
            t += 1
            if done:
                break
    return V

def sarsa(env, alpha, epsilon, gamma, num_episodes):
    Q = np.zeros((env.num_states, env.num_actions))
    Q_list = []
    rewards = []
    for i in range(num_episodes):
        state = env.reset()
        action = epsilon_greedy_policy(env, Q, state, epsilon)
        done = False
        episode_reward = 0
        while not done:
            next_state, reward, done = env.step(action)
            next_action = epsilon_greedy_policy(env, Q, next_state, epsilon)
            Q[state, action] += alpha * (reward + gamma * Q[next_state, next_action] - Q[state, action])
            state = next_state
            action = next_action
            episode_reward += reward
        Q_list.append(np.mean(np.max(Q, axis = 1))) 
        rewards.append(episode_reward)
    return Q, Q_list, rewards

######## Q-Learning TD(0) ########

def q_learning(env, alpha, epsilon, gamma, num_episodes):
    Q = np.zeros((env.num_states, env.num_actions))
    Q_list = []
    returns = []
    for i in range(num_episodes):
        s = env.reset()
        done = False
        episode_reward = 0
        while not done:
            if np.random.uniform() < epsilon:
                a = np.random.choice(env.num_actions)
            else:
                a = np.argmax(Q[s, :])

            s_next, r, done = env.step(a)
            episode_reward += r

            Q[s, a] += alpha * (r + gamma * np.max(Q[s_next, :]) - Q[s, a])

            s = s_next

        returns.append(episode_reward)
        Q_list.append(np.mean(np.max(Q, axis = 1)))

    V = np.max(Q, axis=1)
    policy = np.argmax(Q, axis=1)
    return Q, Q_list, V, policy, returns
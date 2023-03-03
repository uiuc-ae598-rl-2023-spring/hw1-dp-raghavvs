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
    def __init__(self, env, gamma, threshold):
        self.env = env
        self.gamma = gamma
        self.threshold = threshold
        self.values = np.zeros(env.num_states)

    def value_iteration(self):
        delta = np.inf
        while delta > self.threshold:
            delta = 0
            delta_diff = 0
            delta_list = []
            policy_star = []
            for s in range(self.env.num_states):
                v = self.values[s]
                max_action_value = -np.inf
                for a in range(self.env.num_actions):
                    action_value = 0
                    for s1 in range(self.env.num_states):
                        action_value += self.env.p(s1, s, a) * (self.env.r(s, a) + self.gamma * self.values[s1])
                    max_action_value = max(max_action_value, action_value)
                self.values[s] = max_action_value
                policy_star.append(np.argmax(max_action_value))
                delta_diff = abs(v - self.values[s])
                delta = max(delta, abs(v - self.values[s]))
            delta_list.append(delta_diff)
        return delta_list, policy_star

######## SARSA ########

def sarsa(env, alpha=0.1, gamma=0.95, epsilon=0.1, num_episodes=100):
    q_values = [[0.0 for _ in range(env.num_actions)] for _ in range(env.num_states)]
    returns = []
    for episode in range(num_episodes):
        state = env.reset()
        action = None
        done = False
        total_reward = 0
        while not done:
            if random.random() < epsilon:
                action = random.randrange(env.num_actions)
            else:
                action = max(range(env.num_actions), key=lambda a: q_values[state][a])
            next_state, reward, done = env.step(action)
            total_reward += reward
            next_action = None
            if not done:
                if random.random() < epsilon:
                    next_action = random.randrange(env.num_actions)
                else:
                    next_action = max(range(env.num_actions), key=lambda a: q_values[next_state][a])
                td_error = reward + gamma * q_values[next_state][next_action] - q_values[state][action]
            else:
                td_error = reward - q_values[state][action]
            q_values[state][action] += alpha * td_error
            state = next_state
            action = next_action
        returns.append(total_reward)

    return [max(q_values[s]) for s in range(env.num_states)], returns

######## SARSA TD(0) ########

def sarsa_TD(env, alpha=0.1, gamma=0.95, epsilon=0.1, num_episodes=100):
    q_values = [[0.0 for _ in range(env.num_actions)] for _ in range(env.num_states)]
    values = np.zeros(env.num_states)
    returns = []
    for episode in range(num_episodes):
        state = env.reset()
        action = None
        done = False
        total_reward = 0
        while not done:
            if random.random() < epsilon:
                action = random.randrange(env.num_actions)
            else:
                action = max(range(env.num_actions), key=lambda a: q_values[state][a])
            next_state, reward, done = env.step(action)
            total_reward += reward
            next_action = None
            if not done:
                if random.random() < epsilon:
                    next_action = random.randrange(env.num_actions)
                else:
                    next_action = max(range(env.num_actions), key=lambda a: q_values[next_state][a])
                td_target = reward + gamma * values[next_state]
            else:
                td_target = reward
            td_error = td_target - values[state]
            values[state] += alpha * td_error
            state = next_state
            action = next_action
        returns.append(total_reward)

    return values, returns

######## Q-Learning ########

def q_learning(env, num_episodes, alpha, gamma, epsilon):
    Q = np.zeros((env.num_states, env.num_actions))

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

    V = np.max(Q, axis=1)
    policy = np.argmax(Q, axis=1)
    return Q, V, policy, returns

######## Q-Learning TD(0) ########

def q_learning_TD(env, num_episodes, alpha, gamma, epsilon):
    Q = np.zeros((env.num_states, env.num_actions))

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

            td_error = r + gamma * np.max(Q[s_next, :]) - Q[s, a]
            Q[s, a] += alpha * td_error

            s = s_next

        returns.append(episode_reward)

    V = np.max(Q, axis=1)
    policy = np.argmax(Q, axis=1)
    return V, policy, returns
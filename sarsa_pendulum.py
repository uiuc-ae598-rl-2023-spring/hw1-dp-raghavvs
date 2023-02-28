import random
import numpy as np
import matplotlib.pyplot as plt
import discrete_pendulum

pendulum = discrete_pendulum.Pendulum()

def sarsa(pendulum, alpha=0.1, gamma=0.95, epsilon=0.1, num_episodes=100):
    q_values = [[0.0 for _ in range(pendulum.num_actions)] for _ in range(pendulum.num_states)]
    returns = []
    for episode in range(num_episodes):
        state = pendulum.reset()
        action = None
        done = False
        total_reward = 0
        while not done:
            if random.random() < epsilon:
                action = random.randrange(pendulum.num_actions)
            else:
                action = max(range(pendulum.num_actions), key=lambda a: q_values[state][a])
            next_state, reward, done = pendulum.step(action)
            total_reward += reward
            next_action = None
            if not done:
                if random.random() < epsilon:
                    next_action = random.randrange(pendulum.num_actions)
                else:
                    next_action = max(range(pendulum.num_actions), key=lambda a: q_values[next_state][a])
                td_error = reward + gamma * q_values[next_state][next_action] - q_values[state][action]
            else:
                td_error = reward - q_values[state][action]
            q_values[state][action] += alpha * td_error
            state = next_state
            action = next_action
        returns.append(total_reward)

    return [max(q_values[s]) for s in range(pendulum.num_states)], returns

sarsa_returns = sarsa(pendulum, num_episodes=500)

# Plot 1 - Return vs. Episodes
num_episodes = 500
_, returns = sarsa(pendulum, num_episodes=num_episodes)
plt.plot(range(num_episodes), returns)
plt.xlabel('Episode')
plt.ylabel('Return')
plt.title('SARSA Learning Curve')
plt.savefig('figures/pendulum/sarsa_return_vs_episodes.png')
#plt.show()

# Plot 2 - SARSA Learning Curves for Different Epsilon Values
epsilons = [0.1, 0.2, 0.3, 0.4, 0.5]
num_episodes = 10
fig, ax = plt.subplots()
for epsilon in epsilons:
    _, returns = sarsa(pendulum, epsilon=epsilon, num_episodes=num_episodes)
    ax.plot(range(num_episodes), returns, label=f'epsilon={epsilon}')
ax.legend()
ax.set_xlabel('Episode')
ax.set_ylabel('Return')
ax.set_title('SARSA Learning Curves for Different Epsilon Values')
plt.savefig('figures/pendulum/sarsa_learning_curves_epsilon.png')
#plt.show()

# Plot 3 - SARSA Learning Curves for Different Alpha Values 
alphas = [0.1, 0.2, 0.3, 0.4, 0.5]
num_episodes = 10
fig, ax = plt.subplots()
for alpha in alphas:
    _, returns = sarsa(pendulum, alpha=alpha, num_episodes=num_episodes)
    ax.plot(range(num_episodes), returns, label=f'alpha={alpha}')
ax.legend()
ax.set_xlabel('Episode')
ax.set_ylabel('Return')
ax.set_title('SARSA Learning Curves for Different Alpha Values')
plt.savefig('figures/pendulum/sarsa_learning_curves_alpha.png')
#plt.show()
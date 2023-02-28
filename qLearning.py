import numpy as np
import matplotlib.pyplot as plt
import gridworld

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


env = gridworld.GridWorld(hard_version=False) 

# Plot 1 - Return vs. Episodes
Q, V, policy, returns = q_learning(env, num_episodes=500, alpha=0.1, gamma=0.9, epsilon=0.1)
plt.plot(range(len(returns)), returns)
plt.xlabel('Episode')
plt.ylabel('Return')
plt.title('Q-Learning Learning Curve')
plt.savefig('figures/gridworld/ql_return_vs_episodes.png')
#plt.show()

# Plot 2 - Q-Learning Learning Curves for Different Epsilon Values
epsilons = [0.1, 0.2, 0.3, 0.4, 0.5]
num_episodes = 500
fig, ax = plt.subplots()
for epsilon in epsilons:
    _, _, _, returns = q_learning(env, num_episodes=num_episodes, alpha=0.1, gamma=0.9, epsilon=epsilon)
    ax.plot(range(num_episodes), returns, label=f'epsilon={epsilon}')
ax.legend()
ax.set_xlabel('Episode')
ax.set_ylabel('Return')
ax.set_title('Q-Learning Learning Curves for Different Epsilon Values')
plt.savefig('figures/gridworld/ql_learning_curves_epsilon.png')
#plt.show()

# Plot 3 - Q-Learning Learning Curves for Different Alpha Values 
alphas = [0.1, 0.2, 0.3, 0.4, 0.5]
num_episodes = 500
fig, ax = plt.subplots()
for alpha in alphas:
    _, _, _, returns = q_learning(env, num_episodes=num_episodes, alpha=alpha, gamma=0.9, epsilon=0.1)
    ax.plot(range(num_episodes), returns, label=f'alpha={alpha}')
ax.legend()
ax.set_xlabel('Episode')
ax.set_ylabel('Return')
ax.set_title('Q-Learning Learning Curves for Different Alpha Values')
plt.savefig('figures/gridworld/ql_learning_curves_alpha.png')
#plt.show()
import random
import numpy as np
import matplotlib.pyplot as plt

def trajectory(env, policy):
    
    # Simulate until episode is done
    s     = env.reset()
    done  = False
    log = {
        't': [0],
        's': [s],
        'a': [],
        'r': [],
    }
    while not done:
        a = policy[s]
        (s, r, done) = env.step(a)
        log['t'].append(log['t'][-1] + 1)
        log['s'].append(s)
        log['a'].append(a)
        log['r'].append(r)
    return log

def plot_value_function(Q):
    values = np.max(Q, axis=1).reshape((5, 5))
    plt.imshow(values, cmap='cool')
    for i in range(4):
        for j in range(4):
            plt.text(j, i, "{:.2f}".format(values[i, j]), ha="center", va="center", color="w", fontsize=14)
    plt.xticks([])
    plt.yticks([])
    plt.title("Optimal Value Function")

def plot_q_episodes(Q_list, value, v_name):
    episodes = np.arange(len(Q_list)) + 1
    plt.plot(episodes, Q_list, label = v_name + "{}".format(value))

def plot_v_episodes(Q_list, algo_name, x_label , y_label, title, color):
    epsilon = np.arange(len(Q_list)) + 1
    plt.figure(figsize=(10, 8))
    plt.plot(epsilon, Q_list, color, label = algo_name)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    #plt.show()

def trajectory_p(env, policy):
    s = env.reset()
    done = False
    log = {
        't': [0],
        's': [s],
        'a': [],
        'r': [],
        'theta': [env.x[0]],        # agent does not have access to this, but helpful for display
        'thetadot': [env.x[1]],     # agent does not have access to this, but helpful for display
    }

    # Simulate until episode is done
    while not done:
        a = random.randrange(env.num_actions)
        (s, r, done) = env.step(a)
        log['t'].append(log['t'][-1] + 1)
        log['s'].append(s)
        log['a'].append(a)
        log['r'].append(r)
        log['theta'].append(env.x[0])
        log['thetadot'].append(env.x[1])
    return log
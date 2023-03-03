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
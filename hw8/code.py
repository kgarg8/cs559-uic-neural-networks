# Q-learning implementation for 5*5 grid

import numpy as np, matplotlib.pyplot as plt, random, pdb
from tqdm import tqdm

#### Environment ####

# HO 21 22 23 24
# 15 16 17 18 19
# 10 11 12 13 14
#  5  6  7  8  9
#  I  1  2  3 GM

# GM: Gold Mine   (4)
# HO: Home        (20)
#  I: Initial Pos (0)

######################

# environment settings
GRID_SIZE = 5
ACTIONS   = 4                       # R, L, U, D
GOLD_MAX  = 3                       # maximum gold value
STATES    = GRID_SIZE*GRID_SIZE
GOLD_MINE = GRID_SIZE-1             # location of gold mine
HOME      = GRID_SIZE*(GRID_SIZE-1) # location of home

# hyperparameters
num_episodes    = 100
train_timesteps = 5000
test_timesteps  = 500
alpha           = 0.10 # learning rate
gamma           = 0.9  # discount factor
prob            = 0.1  # probability for exploration

# seed
seed = 112
np.random.seed(seed)
random.seed(seed)

# global parameters
Q_table   = np.random.normal(size=(STATES, GOLD_MAX+1, ACTIONS)) # Initialize with Gaussian Distribution values
action_map = {0:'R', 1:'L', 2:'U', 3:'D'}                        # Actions


# returns (reward, next_state, new_gold_val)
def reward(state, action, gold):

    # next step is GOLD MINE
    if (state==GOLD_MINE-1 and action==0) or (state==GOLD_MINE+GRID_SIZE and action==3):
        if gold < GOLD_MAX:
            gold = gold + 1
        return (0, state, gold) # zero reward, stay in the same state

    # next step is HOME
    elif (state==HOME+1 and action==1) or (state==HOME-GRID_SIZE and action==2):
        ret = (gold, state, 0) # reward equals gold and gold gets unloaded
        gold = 0
        return ret # non-zero reward, stay in the same state

    # RIGHT
    elif action == 0:
        if state%GRID_SIZE == GRID_SIZE-1:
            return (0, state, gold) # no way to go
        else:
            return (0, state+1, gold)

    # LEFT
    elif action == 1:
        if state%GRID_SIZE == 0:
            return (0, state, gold) # no way to go
        else:
            return (0, state-1, gold)

    # UP
    elif action == 2:
        if state//GRID_SIZE == GRID_SIZE-1:
            return (0, state, gold) # no way to go
        else:
            return (0, state+GRID_SIZE, gold)

    # DOWN
    else:
        if state//GRID_SIZE == 0:
            return (0, state, gold) # no way to go
        else:
            return (0, state-GRID_SIZE, gold)


def plot(rewards, label):
    if label == 'Test timesteps':
        arr = [i for i in range(test_timesteps)]
    else: # 'Episodes'
        arr = [i for i in range(num_episodes)]
    plt.figure()
    plt.xlabel(label)
    plt.ylabel('Cumulative Reward')
    plt.plot(arr, rewards)
    plt.title('{} vs. Cumulative Reward'.format(label))
    plt.savefig('{} vs. Cumulative Reward_gamma_{}.png'.format(label, gamma))


def episodic_test(episode):
    x = 0; gold = 0; cum_reward = 0 # initial state
    for t in range(test_timesteps):
        a = np.argmax(Q_table[(x, gold)])
        r, y, gold = reward(x, a, gold)
        x = y
        cum_reward += r*(gamma**t)
    print('Episode: {}, Cumulative Reward: {}'.format(episode, cum_reward))
    return cum_reward


def train():
    rewards = []
    for e in range(num_episodes):
        x = 0; gold = 0 # initial state
        for t in range(train_timesteps):
            a = random.random()
            if a < prob:    
                a = random.randint(0, 3)          # exploration
            else:
                a = np.argmax(Q_table[(x, gold)]) # exploitation

            prev_state = (x, gold)
            r, y, gold = reward(x, a, gold)

            a_Qmax = np.argmax(Q_table[(y, gold)])
            Q_table[prev_state][a] = (1-alpha)*Q_table[prev_state][a] + alpha*(r + gamma*Q_table[(y, gold)][a_Qmax])
            
            # logging
            # print('Cur state: {}, Action: {}, New state: {}, Reward: {}, Gold: {}'.format(x, action_map[a], y, r, gold))
            # if r != 0:  print(t)
            
            x = y # save new state
        
        episodic_reward = episodic_test(e)
        rewards.append(episodic_reward)
    plot(rewards, 'Episodes')


def test():
    x = 0; gold = 0; cum_reward = 0 # initial state
    rewards = []
    for t in range(test_timesteps):
        a = np.argmax(Q_table[(x, gold)])
        r, y, gold = reward(x, a, gold)
        
        # logging
        print('Timestep: {},  Cur state: {}, Action: {}, New state: {}, Reward: {}, Gold: {}'.format(t+1, x, action_map[a], y, r, gold))
        # if r != 0:  print(t)

        x = y
        cum_reward += r*(gamma**t)
        rewards.append(cum_reward)

    plot(rewards, 'Test timesteps')


train()
test()
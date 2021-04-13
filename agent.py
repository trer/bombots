import random
import sys
import numpy as np
import pandas as pd
sys.path.append("..")
from environment.bombots import Bombots
from collections import Counter

class AiAgent:
    
    # Feel free to remove the comments from this file, they 
    # can be found in the source on GitHub at any time

    # https://github.com/CogitoNTNU/bombots

    # This is an example agent that can be used as a learning resource or as 
    # a technical base for your own agents. There are few limitations to how 
    # this should be structured, but the functions '__init__' and 'act' need
    # to have the exact parameters that they have in the original source.

    # Action reference:
    # Bombots.NOP    - No Operation
    # Bombots.UP     - Move up 
    # Bombots.DOWN   - Move down
    # Bombots.LEFT   - Move left
    # Bombots.RIGHT  - Move right
    # Bombots.BOMB   - Place bomb

    # State reference: 
    # env_state['agent_pos'] - Coordinates of the controlled bot - (x, y) tuple
    # env_state['enemy_pos'] - Coordinates of the opponent bot   - (x, y) tuple
    # env_state['bomb_pos']  - List of bomb coordinates          - [(x, y), (x, y)] list of tuples
    
    def __init__(self, env):
        self.env = env
        self.agent = QLearningTable()

    def act(self, env_state):
        smart_moves = []

        # Get agent coordinates from dictionary
        x, y = env_state['agent_pos']

        # Combine box map and wall map into one collision matrix (as both are solid)
        solid_map = np.logical_or(self.env.box_map, self.env.wall_map)


        # Check for collisions in neighboring tiles
        if x + 1 in range(0, self.env.w) and solid_map[x + 1][y] == 0: smart_moves.append(Bombots.RIGHT)
        if x - 1 in range(0, self.env.w) and solid_map[x - 1][y] == 0: smart_moves.append(Bombots.LEFT)
        if y + 1 in range(0, self.env.h) and solid_map[x][y + 1] == 0: smart_moves.append(Bombots.DOWN)
        if y - 1 in range(0, self.env.h) and solid_map[x][y - 1] == 0: smart_moves.append(Bombots.UP)

        # If standing on a bomb [Just an example, not used here]
        if env_state['agent_pos'] in env_state['bomb_pos']:
            pass
        
        # If possible, consider bombing
        if env_state['agent_ref'].ammo > 0: 
            smart_moves.append(Bombots.BOMB)

        # Choose randomly among the actions that seem relevant
        if len(smart_moves) > 0:
            action = random.choice(smart_moves) 
        else: action = Bombots.NOP
        
        return action
"""
This part of code is the Q learning brain, which is a brain of the agent.
All decisions are made in here.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""



class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        self.check_state_exist(observation)
        # action selection
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.loc[observation, :]
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

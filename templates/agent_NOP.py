import random
import sys
import numpy as np
sys.path.append("..")
from bombots.environment import Bombots
from collections import Counter

class NOPAgent:
    
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
        self.hasprinted = False
        self.counter = 0

    def act(self, env_state):
        action = Bombots.NOP

        return action






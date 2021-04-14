import random
import sys
import numpy as np

sys.path.append("..")
from bombots.environment import Bombots
from collections import Counter


class BeatNopAgent:

    def __init__(self, env):
        self.env = env
        self.danger_zone = {}
        self.pos = (0, 0)

    def reset(self):
        self.danger_zone = {}

    def find_danger_zone(self, env_state):
        # Reduce timer on blast
        for key in self.danger_zone:
            if self.danger_zone[key] > 0:
                self.danger_zone[key] -= 1

        # find cross where bombs are
        for bomb in env_state['bomb_pos']:
            for i in range(0, self.env.w):
                self.danger_zone[(i, bomb[1])] = 6
            for i in range(0, self.env.h):
                self.danger_zone[(bomb[0], i)] = 6

    def act(self, env_state):

        possible_moves = []

        # Get agent coordinates from dictionary
        x, y = env_state['agent_pos']
        self.pos = (x, y)

        # Combine box map and wall map into one collision matrix (as both are solid)
        solid_map = np.logical_or(self.env.box_map, self.env.wall_map)
        self.env


        # print(f"map of danger_zone: {self.danger_zone}")
        self.find_danger_zone(env_state) # updates map of where bombs are going to explode

        danger_zone = []
        for key in self.danger_zone:
            if self.danger_zone[key] > 0:
                danger_zone.append(key)

        #print(f"real danger_zone: {danger_zone}")

        if (x, y) not in danger_zone:
            if x + 1 in range(0, self.env.w) and solid_map[x + 1][y] == 0 and (x+1, y) not in danger_zone: possible_moves.append(Bombots.RIGHT)
            if x - 1 in range(0, self.env.w) and solid_map[x - 1][y] == 0 and (x-1, y) not in danger_zone: possible_moves.append(Bombots.LEFT)
            if y + 1 in range(0, self.env.h) and solid_map[x][y + 1] == 0 and (x, y+1) not in danger_zone: possible_moves.append(Bombots.DOWN)
            if y - 1 in range(0, self.env.h) and solid_map[x][y - 1] == 0 and (x, y-1) not in danger_zone: possible_moves.append(Bombots.UP)
        else:
            if x + 1 in range(0, self.env.w) and solid_map[x + 1][y] == 0: possible_moves.append(Bombots.RIGHT)
            if x - 1 in range(0, self.env.w) and solid_map[x - 1][y] == 0: possible_moves.append(Bombots.LEFT)
            if y + 1 in range(0, self.env.h) and solid_map[x][y + 1] == 0: possible_moves.append(Bombots.DOWN)
            if y - 1 in range(0, self.env.h) and solid_map[x][y - 1] == 0: possible_moves.append(Bombots.UP)

        # If standing on a bomb [Just an example, not used here]
        if env_state['agent_pos'] in env_state['bomb_pos']:
            pass

        # If possible, consider bombing
        if env_state['agent_ref'].ammo > 0 and len(danger_zone) == 0:
            possible_moves.append(Bombots.BOMB)

        # Choose randomly among the actions that seem relevant
        if len(possible_moves) > 0:
            action = random.choice(possible_moves)
        else:
            action = Bombots.NOP

        if action == Bombots.UP:
            self.pos = (x, y-1)
        if action == Bombots.DOWN:
            self.pos = (x, y+1)
        if action == Bombots.LEFT:
            self.pos = (x-1, y)
        if action == Bombots.RIGHT:
            self.pos = (x+1, y)
        #print(f"pos after action: {self.pos}")

        return action





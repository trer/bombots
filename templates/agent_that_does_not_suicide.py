import random
import sys
import numpy as np

sys.path.append("../..")
from bombots.environment import Bombots
from collections import Counter


class BeatNopAgent:

    def __init__(self, env):
        self.env = env
        self.danger_zone = {}
        self.pos = (0, 0)

    def reset(self):
        self.danger_zone = {}

    def find_danger_zone(self, state):
        """
        me_cord = np.where(state[0] == 1)
        opponent_cord = np.where(state[1] == 1)
        bomb_cords = np.where(state[2] == 1)
        fire_cords = np.where(state[3] == 1)
        crates_cords = np.where(state[4] == 1)
        solid_map_cords = np.where(state[5] == 1)
        """
        # Reduce timer on blast
        for key in self.danger_zone:
            if self.danger_zone[key] > 0:
                self.danger_zone[key] -= 1

        # find cross where bombs are
        lst = np.argwhere(state[2] == 1)

        for bomb in lst:
            if bomb.size == 0:
                break
            for i in range(0, self.env.width):
                self.danger_zone[(i, bomb[1])] = 6
            for i in range(0, self.env.height):
                self.danger_zone[(bomb[0], i)] = 6

    def act(self, state):

        possible_moves = []

        # Get agent coordinates from dictionary
        x, y = env_state['self_pos']

        self.pos = (x, y)

        # Combine box map and wall map into one collision matrix (as both are solid)
        solid_map = np.logical_or(self.env.box_map, self.env.wall_map)


        # print(f"map of danger_zone: {self.danger_zone}")
        self.find_danger_zone(state) # updates map of where bombs are going to explode

        danger_zone = []
        for key in self.danger_zone:
            if self.danger_zone[key] > 0:
                danger_zone.append(key)

        if self.pos not in danger_zone:
            if x + 1 in range(0, self.env.width) and solid_map[x + 1][y] == 0 and (x+1, y) not in danger_zone: possible_moves.append(Bombots.RIGHT)
            if x - 1 in range(0, self.env.width) and solid_map[x - 1][y] == 0 and (x-1, y) not in danger_zone: possible_moves.append(Bombots.LEFT)
            if y + 1 in range(0, self.env.height) and solid_map[x][y + 1] == 0 and (x, y+1) not in danger_zone: possible_moves.append(Bombots.DOWN)
            if y - 1 in range(0, self.env.height) and solid_map[x][y - 1] == 0 and (x, y-1) not in danger_zone: possible_moves.append(Bombots.UP)
        else:
            if x + 1 in range(0, self.env.width) and solid_map[x + 1][y] == 0: possible_moves.append(Bombots.RIGHT)
            if x - 1 in range(0, self.env.width) and solid_map[x - 1][y] == 0: possible_moves.append(Bombots.LEFT)
            if y + 1 in range(0, self.env.height) and solid_map[x][y + 1] == 0: possible_moves.append(Bombots.DOWN)
            if y - 1 in range(0, self.env.height) and solid_map[x][y - 1] == 0: possible_moves.append(Bombots.UP)



        # If possible, consider bombing
        state_dict = self.env.get_state_dict(self.env.bbots[1])
        if state_dict['ammo'] > 0 and len(danger_zone) == 0:
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





import random
import sys
import numpy as np
sys.path.append("..")
from bombots.environment import Bombots
from collections import Counter

import numpy as np
import time


nodes = []
edge_count = 0



class OLDTestAgent:
    
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
    # env_state['self_pos'] - Coordinates of the controlled bot - (x, y) tuple
    # env_state['opponent_pos'] - Coordinates of the opponent bot   - (x, y) tuple
    # env_state['bomb_pos']  - List of bomb coordinates          - [(x, y), (x, y)] list of tuples

    # env.wall_map  -   np-array of walls
    # env.box_map   -   np-array of boxes
    
    def __init__(self, env):
        self.env = env

        self.hasprinted = False
        self.counter = 0

        self.known_bombs = {}
        # self.known_bombs  - Dict of bombs - { "x-y" : [tick, strength, bomb] }

        # MIRRORED WITH ENEMY
        self.agent_strength = 1

        # ENEMY
        self.enemy_strength = 1

    # return [(x1,y1), (x2, y2)]
    def find_path(self, start, end):
        
        walls = self.env.wall_map
        boxes = self.env.box_map

        wx,wy = walls.shape

        path_data = []

        for x in range(wx):
            for y in range(wy):
                if walls[x][y] != 1:
                    if x > 0:
                        dist = 1
                        if boxes[x-1][y] == 1:
                            dist = 33
                        path_data.append([f"{x}-{y}", f"{x-1}-{y}", dist])
                    
                    if x < wx-1:
                        dist = 1
                        if boxes[x+1][y] == 1:
                            dist = 33
                        path_data.append([f"{x}-{y}", f"{x+1}-{y}", dist])

                    if y > 0:
                        dist = 1
                        if boxes[x][y-1] == 1:
                            dist = 33
                        path_data.append([f"{x}-{y}", f"{x}-{y-1}", dist])
                    
                    if y < wy-1:
                        dist = 1
                        if boxes[x][y+1] == 1:
                            dist = 33
                        path_data.append([f"{x}-{y}", f"{x}-{y+1}", dist])


        path_array = solve_dijkstra(path_data, start, end)[1:]
        path_tuples = []
        for tile in path_array:
            x,y = int(tile.split("-")[0]), int(tile.split("-")[1])
            path_tuples.append((x,y))
        
        return path_tuples

    def update_bombs(self, env_state):
        # env_state['self_pos'] - Coordinates of the controlled bot - (x, y) tuple
        # env_state['opponent'] - Coordinates of the opponent bot   - (x, y) tuple
        # env_state['bomb_pos']  - List of bomb coordinates          - [(x, y), (x, y)] list of tuples

        # self.known_bombs  - Dict of bombs - { "x-y" : [tick, strength, bomb] }

        # Tick down any known bombs, remove them if they explode
        rem = []
        for key in self.known_bombs.keys():
            bomb = self.known_bombs[key]
            if bomb[0] > 0:
                bomb[0] -= 1
            else:
                rem.append(key)
        # Remove any that are at 0 (exploded this round)
        for key in rem:
            self.known_bombs.pop(key, None)

        # Keep track of the new bomb position, strength, as well as fuse ticks
        for bomb in env_state['bomb_pos']:
            bomb_string = f"{bomb[0]}-{bomb[1]}"
            if not bomb_string in self.known_bombs.keys():
                if env_state['self_pos'] == bomb:
                    bomb_info = [30, self.agent_strength, bomb]
                else:
                    bomb_info = [30, self.enemy_strength, bomb]
                self.known_bombs[bomb_string] = bomb_info
        
        # Weed out bombs that have exploded preemptively
        rem = []
        for key in self.known_bombs.keys():
            if not self.known_bombs[key][2] in env_state['bomb_pos']:
                rem.append(key)

        for key in rem:
            self.known_bombs.pop(key, None)
        
        # print(self.known_bombs)

    def get_danger_zone(self):
        
        # self.known_bombs  - Dict of bombs - { "x-y" : [tick, strength, bomb] }

        danger_zone = []
        wx,wy = self.env.wall_map.shape

        for key in self.known_bombs.keys():
            bomb_info = self.known_bombs[key]

            # if bomb_info[0] > 8:    #   Limits danger_zone to bombs that are about to go off
            #     break

            x,y = bomb_info[2]

            for i in range(x, x+bomb_info[1]+1):
                if i == wx:
                    break
                if self.env.wall_map[i][y] == 1 or self.env.box_map[i][y] == 1:
                    break
                danger_zone.append((i,y))
                # print(f"Danger to the right of {bomb_info[2]}")

            for i in range(x, x-bomb_info[1]-1, -1):
                if i == -1:
                    break
                if self.env.wall_map[i][y] == 1 or self.env.box_map[i][y] == 1:
                    break
                danger_zone.append((i,y))
                # print(f"Danger to the left of {bomb_info[2]}")

            for i in range(y, y+bomb_info[1]+1):
                if i == wy:
                    break
                if self.env.wall_map[x][i] == 1 or self.env.box_map[x][i] == 1:
                    break
                danger_zone.append((x,i))
                # print(f"Danger to the below of {bomb_info[2]}")

            for i in range(y, y-bomb_info[1]-1, -1):
                if i == -1:
                    break
                if self.env.wall_map[x][i] == 1 or self.env.box_map[x][i] == 1:
                    break
                danger_zone.append((x,i))
                # print(f"Danger to the upper of {bomb_info[2]}")

        return set(danger_zone)

    def find_closest_safe_tile(self, dz, env_state):
        # Find all walkable tiles
        wx,wy = self.env.wall_map.shape

        # Make map of walkable tiles being 1
        walkable = np.zeros([wx,wy])
        for x in range(wx):
            for y in range(wy):
                if self.env.wall_map[x][y] == 0 and self.env.box_map[x][y] == 0:
                    walkable[x][y] = 1
        
        print(walkable)
        # print(self.env.wall_map)
        wx,wy = walkable.shape

        path_data = []
        # Make edges between danger_zone and "border"
        for tile in dz:
            x,y = tile
            if x > 0:
                if walkable[x-1][y] == 1:
                    path_data.append([f"{x}-{y}", f"{x-1}-{y}", 1])
                    print(f"{x-1}-{y} is equal to {walkable[x-1][y]}")
            if x < wx-1:
                if walkable[x+1][y] == 1:
                    path_data.append([f"{x}-{y}", f"{x+1}-{y}", 1])
                    print(f"{x+1}-{y} is equal to {walkable[x+1][y]}")
            if y > 0:
                if walkable[x][y-1] == 1:
                    path_data.append([f"{x}-{y}", f"{x}-{y-1}", 1])
                    print(f"{x}-{y-1} is equal to {walkable[x][y-1]}")
            if y < wy-1:
                if walkable[x][y+1] == 1:
                    path_data.append([f"{x}-{y}", f"{x}-{y+1}", 1])
                    print(f"{x}-{y+1} is equal to {walkable[x][y+1]}")

        # Remove tiles that are already in danger_zone
        for tile in dz:
            if tile in path_data:
                path_data.remove(tile)
        
        # Find closest safe tile
        closest_safe_tile = solve_bf(path_data, f"{env_state['self_pos'][0]}-{env_state['self_pos'][1]}")

        print(f"Closest safe tile: {closest_safe_tile.symbol}")
        print(f"Path: {get_path_array(closest_safe_tile)}")

        return closest_safe_tile

    # start = agent_pos, end = whatever
    def move_towards_position(self, start, end, env_state):
        # start / end = "x-y"
        next_tile = self.find_path(start, end)[0]
        danger_zone = self.get_danger_zone()

        ax, ay = int(start.split("-")[0]), int(start.split("-")[1])


        nx, ny = next_tile

        if (ax, ay) in danger_zone:
            print("IM IN DANGER HAHAHAHAH")
            closest_safe_tile = self.find_closest_safe_tile(danger_zone, env_state)


            next_tile = get_path_array(closest_safe_tile)[1]


            nx, ny = int(next_tile.split("-")[0]), int(next_tile.split("-")[1])

            if nx > ax:
                return Bombots.RIGHT
            if nx < ax:
                return Bombots.LEFT
            if ny > ay:
                return Bombots.DOWN
            if ny < ay:
                return Bombots.UP

        if next_tile in danger_zone:
            return Bombots.NOP

        if self.env.box_map[next_tile[0]][next_tile[1]] == 1:
            return Bombots.BOMB

        if nx > ax:
            return Bombots.RIGHT
        if nx < ax:
            return Bombots.LEFT
        if ny > ay:
            return Bombots.DOWN
        if ny < ay:
            return Bombots.UP

    def act(self, env_state):
        tic = time.perf_counter()
        
        action = Bombots.NOP
        # print(f"self_pos {env_state['self_pos']}")
        # print(f"self_coords: {env_state['self_pos'][0]}-{env_state['self_pos'][1]}")
        # print(f"opponent_pos {env_state['opponent_pos']}")
        action = self.move_towards_position(
            f"{env_state['self_pos'][0]}-{env_state['self_pos'][1]}",
            f"{env_state['opponent_pos'][0][0]}-{env_state['opponent_pos'][0][1]}",
            env_state
            )

        # TODO
        # CODE FOR BRAIN GOES HERE PLEASE

        self.update_bombs(env_state)
        print(f"AGENT POS: {env_state['self_pos']}")
        print(f"DZ: {self.get_danger_zone()}")
        print(f"BOMB POS: {self.known_bombs}")

        toc = time.perf_counter()
        print(f"{toc-tic:0.4f} seconds")
        return action



class node:
    def __init__(self, symbol):
        self.symbol = symbol
        self.edges = []
        self.shortest_distance = float('inf')
        self.shortest_path_via = None

        nodes.append(self)

    def add_edge(self, node, distance):
        edge = [node, distance]
        if not edge in self.edges:
            self.edges.append(edge)

    def update_edges(self):
        for edge in self.edges:
            distance_via = self.shortest_distance + edge[1]
            if distance_via < edge[0].shortest_distance:
                edge[0].shortest_distance = distance_via
                edge[0].shortest_path_via = self

# Couples two nodes
def make_edge(node1, node2, distance):
    node1.add_edge(node2, distance)
    # node2.add_edge(node1, distance)

# Does the heavy lifting
# Just a python implementation of dijkstras shortest path
def dijkstra(start, end):
    global nodes
    queue = []
    path = []

    queue = nodes.copy()
    start.shortest_distance = 0
    queue.sort(key=lambda node: node.shortest_distance)

    while queue[0] != end:
        node = queue[0]
        node.update_edges()
        path.append(queue.pop(0))
        queue.sort(key=lambda node: node.shortest_distance)
    
    # print(print_path(end))
    # print(f"Distance: {end.shortest_distance}")
    return get_path_array(end)

# Literally just prints the path
def print_path(node):
    if node.shortest_path_via == None:
        return f"{node.symbol}"
    else:
        return f"{print_path(node.shortest_path_via)} -> {node.symbol}"

def get_path_array(node):
    if node.shortest_path_via == None:
        return [node.symbol]
    else:
        return get_path_array(node.shortest_path_via) + [node.symbol]

# Does what it says on the tin
def get_node(symbol):
    for node in nodes:
        if node.symbol == symbol:
            return node
    return 0

# Takes a set of edges, as well as start and end nodes
def solve_dijkstra(edges, start, end):
    # Make edges into nodes and couple them
    global nodes
    nodes = []
    for edge in edges:
        a = get_node(edge[0])
        b = get_node(edge[1])

        if a == 0:
            a = node(edge[0])

        if b == 0:
            b = node(edge[1])

        a.add_edge(b, edge[2])
        # b.add_edge(a, edge[2])

    # Solve path
    return dijkstra(get_node(start), get_node(end))

# Takes a set of edges, as well as start and end nodes
def solve_bf(edges, start):
    global edge_count
    edge_count = 0
    # Make edges into nodes and couple them
    for edge in edges:
        a = get_node(edge[0])
        b = get_node(edge[1])

        if a == 0:
            a = node(edge[0])

        if b == 0:
            b = node(edge[1])

        a.add_edge(b, edge[2])
        edge_count += 1

    return bellman_ford(get_node(start))

def bellman_ford(start):
    start.shortest_distance = 0
    
    for i in range(edge_count):
        for node in nodes:
            for edge in node.edges:
                if node.shortest_distance + edge[1] < edge[0].shortest_distance:
                    # print("bnew short found")
                    edge[0].shortest_distance = node.shortest_distance + edge[1]
                    edge[0].shortest_path_via = node

    nodes.sort(key=lambda n: n.shortest_distance)

    # for node in nodes:
        # print(f"Node: {node.symbol}, distance: {node.shortest_distance}, via: {node.shortest_path_via}")

    return nodes[1]
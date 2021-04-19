import random
import sys
import numpy as np
sys.path.append("..")
from bombots.environment import Bombots
from collections import Counter
import time

debug_print = False

class TestAgent:
    
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
        self.known_bombs = {}

        self.last_round_pu_map = []

        self.agent_strength = 1
        self.ammo = 1
        self.enemy_strength = 1
        self.enemy_ammo = 1

    # Finished
    def get_danger_zone(self, env_state):
        """
        Gives you a map of tiles to avoid bc they're in bomb range, or on fucking fire.
        Takes into account bomb strength, as well as boxes and walls

        PARAMS:
            env_state: the env_state given to self.act()
        RETURNS:
            array[(x,y) tuple]: list of tile-tuples to avoid
        """

        if debug_print: print("FINDING")
        danger_zone = []

        solid_map = np.logical_or(self.env.box_map, self.env.wall_map)
        wx, wy = solid_map.shape

        debug = np.zeros((wx,wy))

        # Update base on bomb reach (takes into account walls and boxes)
        for key in self.known_bombs.keys():

            bomb_info = self.known_bombs[key]
            bomb_pos = bomb_info[2]
            bomb_strength = bomb_info[1]
            x,y = bomb_pos

            # Limits danger zone to bombs about to go off. Comment out if it buggy (yes)
            # if bomb_info[0] > 10:
            #     break


            # Searching right
            for x1 in range(x+1, x+1+bomb_strength):
                if x1 == wx:
                    break
                if solid_map[x1][y] == False:
                    danger_zone.append((x1, y))
                    debug[x1,y] = 1
                    if debug_print: print("DANGER RIGHT")
            # Searching left
            for x2 in range(x-1, x-1-bomb_strength, -1):
                if x2 == -1:
                    break
                if solid_map[x2][y] == False:
                    danger_zone.append((x2, y))
                    debug[x2,y] = 1
                    if debug_print: print("DANGER LEFT")
            # Searching down
            for y1 in range(y+1, y+1+bomb_strength):
                if y1 == wy:
                    break
                if solid_map[x][y1] == False:
                    danger_zone.append((x, y1))
                    debug[x,y1] = 1
                    if debug_print: print("DANGER BELOW")
            # Searching up
            for y2 in range(y-1, y-1-bomb_strength, -1):
                if y2 == -1:
                    break
                if solid_map[x][y2] == False:
                    danger_zone.append((x, y2))
                    debug[x,y2] = 1
                    if debug_print: print("DANGER ABOVE")
            # Add bomb itself to the danger zone
            danger_zone.append(bomb_pos)
            debug[x,y] = 1
        
        # Add fire squares
        for x in range(0, wx):
            for y in range(0, wy):
                if self.env.fire_map[x][y] == 1:
                    if (x,y) not in danger_zone:
                        danger_zone.append((x,y))
                        debug[x,y] = 1
                        if debug_print: print("DANGER FIRE")
        debug = np.fliplr(np.rot90(debug, 3))
        if debug_print: print(debug)
        return danger_zone

    # Finished
    def update_bombs(self, env_state):
        """
        Will update self.known_bombs with data from the current env_state.

        PARAMS:
            env_state: the env_state given to self.act()
        """
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

    # Finished
    def update_powerups(self, env_state):
        """
        Compares this rounds PUs with las round's and decides who picked up what.
        Updates agent/opponent ammo and strength counters accordingly, so bomb/danger_zone info is accurate.

        PARAMS:
            env_state: the env_state given to self.act()
        """

        # find difference to last rounds pu map
        for pu in self.last_round_pu_map:
            if pu not in self.env.upers:

                # compare positions to enemy and agent
                # update relevant player
                if (pu.x, pu.y) == env_state['self_pos']:
                    if pu.upgrade_type == 0:
                        self.ammo += 1
                    else:
                        self.agent_strength += 1
                else:
                    if pu.upgrade_type == 0:
                        self.enemy_ammo += 1
                    else:
                        self.enemy_strength += 1

        # update last round map
        self.last_round_pu_map = self.env.upers.copy()

        pass

    # Finished
    def get_shortest_path_to(self, env_state, destination):
        """
        Will find the shortest path from current position to the given destination.

        PARAMS:
            env_state: the env_state given to self.act()
            destination (string) / (tuple): string of the destination in the form 'x-y' or tuple in the form (x, y)
        RETURNS:
            array[(x,y)]: ordered array of the coord-tuples to visit
        """

        if type(destination) == tuple:
            destination = f"{destination[0]}-{destination[1]}"

        edge_data = []
        
        wx, wy = self.env.wall_map.shape

        # Turn playing field into digraph
        for x in range(0, wx):
            for y in range(0, wy):
                if self.env.wall_map[x][y] == 0:

                    # Edge to tile above
                    if x > 0:
                        if self.env.wall_map[x-1][y] == 0:
                            distance = 1
                            if self.env.box_map[x-1][y] == 1:
                                distance = 33
                            edge_data.append([f"{x}-{y}", f"{x-1}-{y}", distance])
                    
                    # Edge to tile below
                    if x < wx-1:
                        if self.env.wall_map[x+1][y] == 0:
                            distance = 1
                            if self.env.box_map[x+1][y] == 1:
                                distance = 33
                            edge_data.append([f"{x}-{y}", f"{x+1}-{y}", distance])

                    # Edge to tile left
                    if y > 0:
                        if self.env.wall_map[x][y-1] == 0:
                            distance = 1
                            if self.env.box_map[x][y-1] == 1:
                                distance = 33
                            edge_data.append([f"{x}-{y}", f"{x}-{y-1}", distance])

                    # Edge to tile right
                    if y < wy-1:
                        if self.env.wall_map[x][y+1] == 0:
                            distance = 1
                            if self.env.box_map[x][y+1] == 1:
                                distance = 33
                            edge_data.append([f"{x}-{y}", f"{x}-{y+1}", distance])

        nodes = make_nodes(edge_data)

        agent_pos_string = f"{env_state['self_pos'][0]}-{env_state['self_pos'][1]}"
        start = get_node(nodes, agent_pos_string)
        end = get_node(nodes, destination)
        path_nodes = dijkstra(nodes, start, end)
        path_coords = []

        for node in path_nodes:
            path_coords.append(node.get_coordinates())

        return path_coords

    # Finished
    def get_nearest_safe_tile(self, env_state):
        """
        Finds the closest safe tile based on your current position.

        PARAMS:
            env_state: the env_state given to self.act()
        RETURNS:
            (x, y) (int-tuple): coord-tuple of the closest safe nodet o stand on
        """
        danger_zone = self.get_danger_zone(env_state)
        solid_map = np.logical_or(self.env.box_map, self.env.wall_map)

        wx, wy = solid_map.shape

        edge_data = []

        for tile in danger_zone:
            x, y = tile
            # print(f"Danger zone tile: {tile}")

            if x > 0:
                if solid_map[x-1][y] == False:
                    edge_data.append([f"{x}-{y}", f"{x-1}-{y}", 1])
            if x < wx-1:
                if solid_map[x+1][y] == False:
                    edge_data.append([f"{x}-{y}", f"{x+1}-{y}", 1])
            if y > 0:
                if solid_map[x][y-1] == False:
                    edge_data.append([f"{x}-{y}", f"{x}-{y-1}", 1])
            if y < wy-1:
                if solid_map[x][y+1] == False:
                    edge_data.append([f"{x}-{y}", f"{x}-{y+1}", 1])

            # print(f"EDGE DATA this round: {edge_data}")

        # print(f"EDGE DATA: {edge_data}")
        nodes = make_nodes(edge_data)

        ax, ay = env_state['self_pos']
        agent_pos_string = f"{ax}-{ay}"
        start = get_node(nodes, agent_pos_string)

        # debug
        if type(start) != node:
            print(f"start: {start}")
            print(f"pos string: {agent_pos_string}")
            print(f"nodes: {nodes}")
        # debug

        close_nodes = bellman_ford(nodes, len(edge_data), start)

        for tile in danger_zone:
            tile_node = get_node(nodes, f"{tile[0]}-{tile[1]}")
            if tile_node in close_nodes:
                close_nodes.remove(tile_node)

        return close_nodes[0].get_coordinates()

    # Finished
    def get_movement_direction(self, agent_pos, next_tile):
        """
        Returns the action required to reach next tile.

        PARAMS:
            agent_pos (x,y) tuple: tuple of agent's current position
            next_tile (x,y) tuple: tuple of next tile to walk to
        RETURNS:
            action: Bombots action
        """

        x, y = agent_pos
        nx, ny = next_tile

        if x < nx:
            return Bombots.RIGHT
        if x > nx:
            return Bombots.LEFT
        if y < ny:
            return Bombots.DOWN
        if y > ny:
            return Bombots.UP
        
        return Bombots.NOP

    def act(self, env_state):

        tic = time.perf_counter()


        if debug_print: print("\n\nPREINTING MY BUILLSHIT HERE")

        if debug_print: print("UPDATEING")

        # SETUP START

        # SETTING UP USEFUL INFORMATION
        # update bombs
        self.update_bombs(env_state)
        # get dangerzone
        danger_zone = self.get_danger_zone(env_state)
        # get agent pos
        agent_pos = env_state['self_pos']
        enemy_pos = env_state['opponent_pos']
        # update powerups
        self.update_powerups(env_state)

        # Agent is on opponent, and there is a bomb

        # get path to enemy
        enemy_pos_string = f"{enemy_pos[0][0]}-{enemy_pos[0][1]}"
        path_to_enemy = self.get_shortest_path_to(env_state, enemy_pos_string)

        if len(path_to_enemy) > 1:
            next_tile = path_to_enemy[1]
        else:
            next_tile = path_to_enemy[0]
        nx, ny = next_tile


        if debug_print: print(f"MY POSITION: {agent_pos}")
        if debug_print: print(f"NT POSITION: {next_tile}")

        # SETUP END
        


        # DECISION MAKING

        # if agent in dangerzone:
            # if enemy_path.next == safe:
                # move towards enemy
            # else:
                # move towards safe
        # else:
            # if enemy_path.next in danger_zone:
                # NOP
            # elif enemy_path.next == box:
                # place bomb
            # else:
                # move towards enemy
        if debug_print: print("MAKE DECISION")

        # if agent on opponent and no bomb
        if agent_pos in enemy_pos and agent_pos not in danger_zone:
            action = Bombots.BOMB
        elif agent_pos in danger_zone:
            if next_tile not in danger_zone and self.env.box_map[nx][ny] != 1:
                if debug_print: print("move towards enemy")
                action = self.get_movement_direction(agent_pos, next_tile)
            else:
                if debug_print: print("move towards safety")
                next_safe_tile = self.get_nearest_safe_tile(env_state)
                if debug_print: print(f"Nearest safe tile is {next_safe_tile}")
                path_to_nearest_safe_tile = self.get_shortest_path_to(env_state, next_safe_tile)
                next_tile_towards_safety = path_to_nearest_safe_tile[1]
                action = self.get_movement_direction(agent_pos, next_tile_towards_safety)
                pass
        else:
            if next_tile in danger_zone:
                if debug_print: print("dont move")
                action = Bombots.NOP
            elif self.env.box_map[nx][ny] == 1:
                if agent_pos in env_state['bomb_pos']:
                    if debug_print: print("bomb here alreadym, getting da fuck out")
                    next_safe_tile = self.get_nearest_safe_tile(env_state)
                    action = self.get_movement_direction(agent_pos, next_safe_tile)
                else:
                    if debug_print: print("bomb")
                    action = Bombots.BOMB
            else:
                action = self.get_movement_direction(agent_pos, next_tile)
                if debug_print: print("move towards enemy")

    

        # DEBUG SHIDD

        if debug_print: print("\n")

        # self.known_bombs["1-1"] = [30, 2, (1,1)]

        if debug_print: print(f"Bombs: {self.known_bombs}")
        # danger_zone = self.get_danger_zone(env_state)
        # print(f"DAnger cONE: {danger_zone}")
        # if len(danger_zone) > 0:
        #     print(f"Closest safe node: {self.get_nearest_safe_tile(env_state)}")

        # self.get_shortest_path_to(env_state, "9-9")

        # print("FIRE MAP")
        # print(self.env.fire_map)

        # action = Bombots.NOP


        toc = time.perf_counter()
        # print(f"{toc-tic:0.4f} seconds")
        return action


# DEBUGGING SHIDD
def print_path(node):
    if node.shortest_path_via == None:
        return f"{node.symbol}"
    else:
        return f"{print_path(node.shortest_path_via)} -> {node.symbol}"

# PATHFINDING SHIT DOWN HERE NO TOUCHIE >:^(
class node:
    def __init__(self, symbol):
        self.symbol = symbol
        self.edges = []
        self.shortest_distance = float('inf')
        self.shortest_path_via = None

    # Adds another node as a weighted edge
    def add_edge(self, node, distance):
        self.edges.append([node, distance])

    # Checks every node it has an edge to, and updates it if neccessary
    def update_edges(self):
        for edge in self.edges:
            distance_via = self.shortest_distance + edge[1]
            if distance_via < edge[0].shortest_distance:
                edge[0].shortest_distance = distance_via
                edge[0].shortest_path_via = self

    def get_coordinates(self):
        x, y = int(self.symbol.split("-")[0]), int(self.symbol.split("-")[1])
        return (x, y)

def get_node(nodes, symbol):
    """
    Searches "nodes" for node with symbol "symbol" and returns it if found.

    PARAMS:\n
        nodes (array): array of nodes to search from
        symbol (str): string to search matches for
    RETURNS:\n
        node: if match is found
        None: if no match found
    """

    for node in nodes:
        if node.symbol == symbol:
            return node
    return None

def make_nodes(edge_data, *args):
    """
    Takes an array of edges and makes them into node objects.

    PARAMS:
        edge_data (arr): array of edges with format [start_node (str), end_node (str), distance (int)]
        *args (boolean): True if you want digraph, False if not (default is True) Can save time when entering edges by hand.
        *args (array[str]): array of symbols to use for nodes that may not have edges and are not included in "edge_data"
    RETURNS:
        array: array of the nodes that it created
    """

    nodes = []

    # Decide if digraph or not
    if len(args) > 0:
        digraph = args[0]
    else:
        digraph = True

    # Fill in empty nodes
    if len(args) > 1:
        for symbol in args[1]:
            nodes.append(node(symbol))

    # Make edges into nodes and couple them
    for edge in edge_data:
        node1 = get_node(nodes, edge[0])
        node2 = get_node(nodes, edge[1])

        if node1 == None:
            node1 = node(edge[0])

        if node2 == None:
            node2 = node(edge[1])

        node1.add_edge(node2, edge[2])
        if not digraph: node2.add_edge(node1, edge[2])  # REMOVE THIS IF YOU WANT DIGRAPH 2/2
    
        if node1 not in nodes: nodes.append(node1)
        if node2 not in nodes: nodes.append(node2)

    return nodes

def get_path_array(node):
    """
    Takes an end node and gives you every node (in order) for the shortest path to it.

    PARAMS:
        node (node): end node
    
    RETURNS:
        array[nodes]: every note you need to visit (in order)
    """
    if node.shortest_path_via == None:
        return [node]
    else:
        return get_path_array(node.shortest_path_via) + [node]

def dijkstra(nodes, start, end):
    """
    Finds the fastest way from "start" to "end" (usually what dijkstra does).

    PARAMS:
        nodes (array: array of nodes
        start (node): start of path
        end (node): end of path

    RETURNS
        array[node]: path of nodes from "start" to "end" (inclusive) if one is found
        None: if no path is found
    """
    queue = []
    path = []

    # Setup
    queue = nodes.copy()
    start.shortest_distance = 0
    queue.sort(key=lambda node: node.shortest_distance)

    # Exploration loop
    while queue[0] != end:
        node = queue[0]
        node.update_edges()
        path.append(queue.pop(0))
        queue.sort(key=lambda node: node.shortest_distance)
    
    # Test if there actually was a path found
    if end.shortest_distance == float('inf'):
        print("End has not been found")
        return None

    return get_path_array(end)

def bellman_ford(nodes, edge_count, start):
    """
    Takes an array of nodes, and finds the shortest distance from "start" to each of them

    PARAMS:
        nodes (array[node]): array of the relevant nodes
        edge_count (int): number of edges (not nodes)
        start (node): starting node to search from
    
    RETURNS:
        array[node]: array of the nodes sorted by ascending distance to "start"
    """

    start.shortest_distance = 0
    
    for i in range(edge_count):
        for node in nodes:
            for edge in node.edges:
                if node.shortest_distance + edge[1] < edge[0].shortest_distance:
                    edge[0].shortest_distance = node.shortest_distance + edge[1]
                    edge[0].shortest_path_via = node

    nodes.sort(key=lambda x: x.shortest_distance)

    return nodes
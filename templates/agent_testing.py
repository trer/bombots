import random
import sys
import numpy as np
sys.path.append("..")
from bombots.environment import Bombots
from collections import Counter
import time
import copy

debug_print = False
has_printed = False

# TODO:
# Make fires scarier than bombs (bomb less scary the more time till explosion).
# don't corner yourself
# set trap - almost done not quite tested since the FSM is kinda brookie
# Fix FSM
# chain reactions?
# use more bombs


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
        # self.known_bombs  - Dict of bombs - {"x-y":[tick, strength, bomb]}
        self.known_bombs = {}
        self.enemy_pos = []

        self.last_round_pu_map = []

        self.agent_strength = 1
        self.ammo = 1
        self.enemy_strength = 1
        self.enemy_ammo = 1

    def reset(self):
        self.known_bombs = {}
        self.enemy_pos = []

        self.last_round_pu_map = []

        self.agent_strength = 1
        self.ammo = 1
        self.enemy_strength = 1
        self.enemy_ammo = 1

    # Finished
    def get_advanced_danger_zone(self, env_state, *args):
        """
        Makes a list of every tile affected by bombs, and when it will become fire (takes into account possible chain reactions)
        PARAMS:
            env_state: the env_state given to self.act()
            *args[0]: a custom known_bombs dictionairy. Can be ommited
        RETURNS:
            dict[(x, y)] = (int): lookup-dictionairy which returns the ticks_until_detonation of any given tile inside the danger_zone. 0 if it is currently on fire (bad)
        """
        if len(args) == 1:
            bomb_dict = args[0]
        else:
            bomb_dict = self.known_bombs
        

        bomb_affect = {}
        bombs = []
        # Get the reach of every bomb
        for key in bomb_dict:
            bomb_info = bomb_dict[key]
            bomb_affect[bomb_info[2]] = self.chain_reaction(env_state, bomb_info, bomb_dict)
            bombs.append(bomb_info)

        # Sort bombs by ascending ticks
        bombs.sort(key=lambda x: x[0])

        # Group into seperate danger_zones based on reach and order of explosion
        ticks_until_detonation = {}
        exploded = []
        for bomb in bombs:
            if bomb not in exploded:
                exploded.append(bomb)
                # Adds itself to the dics
                temp_dict = {bomb[2]: bomb}
                # Look up bombs affected by this bomb
                for affected_bomb in bomb_affect[bomb[2]]:
                    # Adds every bomb affected by it to the dict
                    temp_dict[affected_bomb[2]] = bomb_dict[get_coord_string(affected_bomb[2])]
                    exploded.append(affected_bomb)

                # Get danger zone of bomb set
                # danger_zone_set = self.debug_get_danger_zone(env_state, temp_dict)
                danger_zone_set = self.get_danger_zone(env_state, temp_dict)
                for tile in danger_zone_set:
                    if tile not in ticks_until_detonation.keys():
                        ticks_until_detonation[tile] = bomb[0]

        wx, wy = self.env.wall_map.shape
        # Add actual fiery tiles
        for x in range(0, wx):
            for y in range(0, wy):
                if self.env.fire_map[x][y] == 1:
                    ticks_until_detonation[(x,y)] = 0

        # Return dictionairy where dangerous tiles are key, and result is ticks until fire
        return ticks_until_detonation

    # Finished
    def chain_reaction(self, env_state, bomb_info, bomb_dict_original):
        """
        Esencially just a help-function for get_advanced_danger_zone. Recursively checks which bombs can be triggered by which other bombs.
        PARAMS:
            env_state: the env_state given to self.act()
            bomb_info (arr[int, int, (x,y)]): a bomb_info array as found in self.known_bombs
            bomb_dict_original: the bomb_dictionairy to be used.
        RETURNS:
            arr[(x,y)]: list of the positions of every bomb that this bomb can reach / affect (through chain reaction or otherwise)
        """

        bomb_dict = bomb_dict_original.copy() # Make backup
        
        # Find the reach of the bomb, remove it from the list
        bomb_name = get_coord_string(bomb_info[2])
        temp = {bomb_name: bomb_info}
        # bomb_reach = self.debug_get_danger_zone(env_state, temp)
        bomb_reach = self.get_danger_zone(env_state, temp)
        del bomb_dict[bomb_name]

        can_reach = []
        # Check for bombs that it will reach
        for key in bomb_dict.keys():
            bomb = bomb_dict[key]
            if bomb[2] in bomb_reach:
                # If it can reach the bomb, add to the list
                can_reach.append(bomb)
                # temp_dic_for_recursive_call = bomb_dict.copy()
                
                # Check all the bombs that this bomb can reach
                recursion_reach = self.chain_reaction(env_state, bomb, bomb_dict)

                # Add every bomb that it can reach to the list as well (will unwrap the array it gets passed down)
                for bomb_in_recursive_reach in recursion_reach:
                    can_reach.append(bomb_in_recursive_reach)
        
        return can_reach

    # Just danger_zone but minus the random boxes. Used for testing dont worry about it
    def debug_get_danger_zone(self, env_state, *args):
        """
        Gives you a map of tiles to avoid bc they're in bomb range, or on fucking fire.
        Takes into account bomb strength, as well as boxes and walls

        PARAMS:
            env_state: the env_state given to self.act()
            *args[0] dict{"x-y":[tick, strength, bomb]}: dictionairy of bombs as seen in self.known_bombs. Will override call to self.knwon_bombs
        RETURNS:
            array[(x,y) tuple]: list of tile-tuples to avoid
        """

        if debug_print: print("FINDING")
        danger_zone = []

        # solid_map = np.logical_or(self.env.box_map, self.env.wall_map)
        solid_map = np.array([[ True, False,  True, False,  True, False,  True, False,  True,
        False,  True],
       [False, False, False, False, False, False, False, False, False,
        False, False],
       [ True, False,  True, False,  True, False,  True, False,  True,
        False,  True],
       [False, False, False, False, False, False, False, False, False,
        False, False],
       [ True, False,  True, False,  True, False,  True, False,  True,
        False,  True],
       [False, False, False, False, False, False, False, False, False,
        False, False],
       [ True, False,  True, False,  True, False,  True, False,  True,
        False,  True],
       [False, False, False, False, False, False, False, False, False,
        False, False],
       [ True, False,  True, False,  True, False,  True, False,  True,
        False,  True],
       [False, False, False, False, False, False, False, False, False,
        False, False],
       [ True, False,  True, False,  True, False,  True, False,  True,
        False,  True]])

        wx, wy = solid_map.shape

        debug = np.zeros((wx,wy))

        if len(args) == 1:
            bomb_dict = args[0]
        else:
            bomb_dict = self.known_bombs.copy()

        # Update base on bomb reach (takes into account walls and boxes)
        for key in bomb_dict.keys():

            bomb_info = bomb_dict[key]
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
                else:
                    break
            # Searching left
            for x2 in range(x-1, x-1-bomb_strength, -1):
                if x2 == -1:
                    break
                if solid_map[x2][y] == False:
                    danger_zone.append((x2, y))
                    debug[x2,y] = 1
                    if debug_print: print("DANGER LEFT")
                else:
                    break
            # Searching down
            for y1 in range(y+1, y+1+bomb_strength):
                if y1 == wy:
                    break
                if solid_map[x][y1] == False:
                    danger_zone.append((x, y1))
                    debug[x,y1] = 1
                    if debug_print: print("DANGER BELOW")
                else:
                    break
            # Searching up
            for y2 in range(y-1, y-1-bomb_strength, -1):
                if y2 == -1:
                    break
                if solid_map[x][y2] == False:
                    danger_zone.append((x, y2))
                    debug[x,y2] = 1
                    if debug_print: print("DANGER ABOVE")
                else:
                    break
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
    def update_bombs(self, env_state, extra_bombs_lst=[], bomb_str=None):
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

        bombs_lst = extra_bombs_lst + env_state['bomb_pos']

        # Keep track of the new bomb position, strength, as well as fuse ticks
        for bomb in bombs_lst:
            bomb_string = f"{bomb[0]}-{bomb[1]}"
            if not bomb_string in self.known_bombs.keys():
                if bomb_str is None:
                    if env_state['self_pos'] == bomb:
                        bomb_info = [30, self.agent_strength, bomb]
                    else:
                        bomb_info = [30, self.enemy_strength, bomb]
                else:
                    bomb_info = [30, bomb_str, bomb]
                self.known_bombs[bomb_string] = bomb_info

        """
        # Weed out bombs that have exploded preemptively
        rem = []
        for key in self.known_bombs.keys():
            if not self.known_bombs[key][2] in env_state['bomb_pos']:
                rem.append(key)

        for key in rem:
            self.known_bombs.pop(key, None)
        """


    # Finished
    def get_danger_zone(self, env_state, *args):
        """
        Gives you a map of tiles to avoid bc they're in bomb range, or on fucking fire.
        Takes into account bomb strength, as well as boxes and walls

        PARAMS:
            env_state: the env_state given to self.act()
            *args[0] dict{"x-y":[tick, strength, bomb]}: dictionairy of bombs as seen in self.known_bombs. Will override call to self.knwon_bombs
        RETURNS:
            array[(x,y) tuple]: list of tile-tuples to avoid
        """

        if debug_print: print("FINDING")
        danger_zone = []

        solid_map = np.logical_or(self.env.box_map, self.env.wall_map)
        wx, wy = solid_map.shape

        debug = np.zeros((wx,wy))

        if len(args) == 1:
            bomb_dict = args[0]
        else:
            bomb_dict = self.known_bombs.copy()


        # Update base on bomb reach (takes into account walls and boxes)
        for key in bomb_dict.keys():

            bomb_info = bomb_dict[key]
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
                else:
                    break
            # Searching left
            for x2 in range(x-1, x-1-bomb_strength, -1):
                if x2 == -1:
                    break
                if solid_map[x2][y] == False:
                    danger_zone.append((x2, y))
                    debug[x2,y] = 1
                    if debug_print: print("DANGER LEFT")
                else:
                    break
            # Searching down
            for y1 in range(y+1, y+1+bomb_strength):
                if y1 == wy:
                    break
                if solid_map[x][y1] == False:
                    danger_zone.append((x, y1))
                    debug[x,y1] = 1
                    if debug_print: print("DANGER BELOW")
                else:
                    break
            # Searching up
            for y2 in range(y-1, y-1-bomb_strength, -1):
                if y2 == -1:
                    break
                if solid_map[x][y2] == False:
                    danger_zone.append((x, y2))
                    debug[x,y2] = 1
                    if debug_print: print("DANGER ABOVE")
                else:
                    break
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

    # Finished, I think (if bombs start behaving weird it is probably this ones fault
    def check_place_bomb(self, env_state, bomb_pos=None, agent_pos=None):
        """Checks wheter it is safe to place a bomb.
        Copys env_state
        places a bomb at players position
        Checks for safe_space
        :param
            env_state state of the map
            me: default=true, if set to False check for enemy
            bomb_pos: (x, y) position of the bomb to be placed
        :return
            int: number of moves to safety, returns -1 if there are no safe tiles.
        """
        bombs = {}
        for key in self.known_bombs.keys():
            bomb = self.known_bombs[key]
            bombs[key] = bomb.copy()
        if bomb_pos is None:
            bomb_pos = [env_state['self_pos']]
        if agent_pos is None:
            agent_pos = env_state['self_pos']
        # bombs[pos] = [30, self.agent_strength, pos]
        self.update_bombs(env_state, bomb_pos, bomb_str=self.agent_strength)
        safe_tile = self.get_nearest_safe_tile(env_state, agent_pos=agent_pos)
        self.known_bombs = bombs


        if safe_tile is None:
            print("did not place a bomb cause it would mean death.")
            return -1
        #for pos in self.get_shortest_path_to(state, safe_tile):
        #    if pos in danger_zone:
        #        print("did not place a bomb because it was scary")
        #        return False
        _, cost = self.get_shortest_path_to(env_state, safe_tile, start=agent_pos)
        return cost

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
                if (pu.pos_x, pu.pos_y) == env_state['self_pos']:
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
    def get_shortest_path_to(self, env_state, destination, me=True, start=None):
        """
        Will find the shortest path from current position to the given destination.

        PARAMS:
            env_state: the env_state given to self.act()
            destination (string) / (tuple): string of the destination in the form 'x-y' or tuple in the form (x, y)
        RETURNS:
            array[(x,y)]: ordered array of the coord-tuples to visit
            int: the weighted distance to the destination tile
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
        if start is None:
            if me:
                agent_pos_string = f"{env_state['self_pos'][0]}-{env_state['self_pos'][1]}"
            else:
                agent_pos_string = f"{env_state['opponent_pos'][0][0]}-{env_state['opponent_pos'][0][1]}"
        else:
            agent_pos_string = f"{start[0]}-{start[1]}"
        start = get_node(nodes, agent_pos_string)
        end = get_node(nodes, destination)
        path_nodes = dijkstra(nodes, start, end)
        path_coords = []

        for node in path_nodes:
            path_coords.append(node.get_coordinates())

        return path_coords, end.shortest_distance

    # Finished
    def get_nearest_safe_tile(self, env_state, agent_pos=None):
        """
        Finds the closest safe tile based on your current position.

        PARAMS:
            env_state: the env_state given to self.act()
            me: boolean value that decides whether to find for yourself or your opponent, default=True
        RETURNS:
            (x, y) (int-tuple): coord-tuple of the closest safe nodet o stand on
            None: if there are no safe tiles
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
        if agent_pos is None:
            ax, ay = env_state['self_pos']
        else:
            ax, ay = agent_pos[0], agent_pos[1]

        if (ax, ay) not in danger_zone:
            return tuple([ax, ay])

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
        if len(close_nodes) == 0:
            return None
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

    # Finished
    def get_closest_upgrade(self, env_state):
        """ Uses get_shortest_path_to on all upgrades on the map and returns the pos of the closest one.
        :param: the state of the current map
        :return path to upgrade with lowest weight (aka the one it can get to the fastest)
        array[(x,y)]: ordered array of the coord-tuples to visit
        if there are no upgrades left on the map it will return None
        """
        shortest_path = []
        shortest_distance = 10000000
        path_set = False
        for pu in self.env.upers:
            path, temp_distance = self.get_shortest_path_to(env_state, (pu.pos_x, pu.pos_y))
            if not path_set or shortest_distance > temp_distance:
                shortest_path = path
                shortest_distance = temp_distance
                path_set = True
        if path_set:
            return shortest_path
        return None

    # TOO SLOW, like a little to slow.
    def corner_enemy(self, env_state, solid_map):
        """This function looks at the current state and tries to find a position to place a bomb such that the enemy
         will have as few places to go as possible.
         :param
            env_state the current state
            solid_map the map that shows combination of boxes (unbreakable) and creates (breakable)
         :returns
            pos (x, y) to place a bomb so as to limit enemy movement
            int spaces enemy has to move to reach safety
            int cost number of turns predicted to achieve this action.

        :returns None if there are no reachable spaces that will limit enemy movement.

         Specifics: position much be reachable without placing bombs.
        """
        path = None
        safety = self.get_nearest_safe_tile(env_state, env_state['opponent_pos'][0])
        our_cost = 0
        if safety is None:
            return env_state['self_pos'], -1, 0
        opp_tile, opp_base_cost = self.get_shortest_path_to(env_state, safety, env_state['opponent_pos'][0])
        for x in range(self.env.width):
            y = env_state['opponent_pos'][0][1]
            if solid_map[x][y] == 1:
                continue
            tmp_tile, tmp_cost = self.get_shortest_path_to(env_state, (x, y))
            if tmp_cost < 30:
                cost = self.check_place_bomb(env_state, bomb_pos=[(x, y)], agent_pos=env_state['opponent_pos'][0])
                if cost == -1:
                    return tmp_tile, cost, tmp_cost
                if cost - tmp_cost > opp_base_cost:
                    path = tmp_tile
                    opp_base_cost = cost
                    our_cost = tmp_cost
        for y in range(self.env.height):
            x = env_state['opponent_pos'][0][0]
            if solid_map[x][y] == 1:
                continue
            tmp_tile, tmp_cost = self.get_shortest_path_to(env_state, (x, y))
            if tmp_cost < 30:
                cost = self.check_place_bomb(env_state, bomb_pos=[(x, y)], agent_pos=env_state['opponent_pos'][0])
                if cost == -1:
                    return tmp_tile, cost, tmp_cost
                if cost > opp_base_cost:
                    path = tmp_tile
                    opp_base_cost = cost
                    our_cost = tmp_cost

        return path, opp_base_cost, our_cost

    def find_closest_tile_that_explodes_on_tile(self, env_state, start_tile, tile, solid_map):
        que = [start_tile]
        index = 0
        found_tile = False
        check_tile = None
        while index < len(que):
            check_tile = que[index]
            if self.check_place_bomb(env_state, bomb_pos=[check_tile], agent_pos=tile) != 0:
                found_tile = True
                break
            upx, upy = check_tile[0], check_tile[1] - 1
            if 11 > upx >= 0 and 11 > upy >= 0:
                if (upx, upy) not in que and solid_map[upx][upy] == 0:
                    que.append((upx, upy))

            downx, downy = check_tile[0], check_tile[1] + 1
            if 11 > downx >= 0 and 11 > downy >= 0:
                if (downx, downy) not in que and solid_map[downx][downy] == 0:
                    que.append((downx, downy))

            leftx, lefty = check_tile[0] - 1, check_tile[1]
            if 11 > leftx >= 0 and 11 > lefty >= 0:
                if (leftx, lefty) not in que and solid_map[leftx][lefty] == 0:
                    que.append((leftx, lefty))

            rightx, righty = check_tile[0] + 1, check_tile[1]
            if 11 > rightx >= 0 and 11 > righty >= 0:
                if (rightx, righty) not in que and solid_map[rightx][righty] == 0:
                    que.append((rightx, righty))

            index += 1
        if found_tile:
            return check_tile
        else:
            return None

    def act(self, env_state):

        # DEBUG SHIDDDD
        # test_dict = {"1-1": [30, 1, (1,1)],"3-1": [12, 2, (3,1)],"3-4": [10, 3, (3,4)],"3-7": [8, 1, (3,7)]}
        # chain_result = []
        # for key in test_dict.keys():
        #     bomb_info = test_dict[key]
        #     chain_result.append(self.chain_reaction(env_state, bomb_info, test_dict))
        # print(chain_result)
        # adz = self.get_advanced_danger_zone(env_state, self.get_danger_zone(env_state, test_dict), test_dict)
        # print(adz)
        # print("\n\n")

        tic = time.perf_counter()

        if debug_print: print("\n\nPREINTING MY BUILLSHIT HERE")

        if debug_print: print("UPDATEING")

        # SETUP START

        # SETTING UP USEFUL INFORMATION
        # update bombs
        self.update_bombs(env_state)
        # get dangerzone
        danger_zone = self.get_danger_zone(env_state).copy()
        advanced_danger_zone = self.get_advanced_danger_zone(env_state)
        # get agent pos
        agent_pos = env_state['self_pos']
        enemy_pos = env_state['opponent_pos']
        self.enemy_pos.append(enemy_pos)
        # update powerups
        self.update_powerups(env_state)
        solid_map = np.logical_or(self.env.box_map, self.env.wall_map)

        # Where to place bomb
        if debug_print: print(f"known bombs before, {self.known_bombs}")
        bomb_path, enemy_cost, cost = self.corner_enemy(env_state, solid_map)
        if debug_print: print(f"known bombs after, {self.known_bombs}")

        # Agent is on opponent, and there is a bomb

        # get path to enemy
        enemy_pos_string = f"{enemy_pos[0][0]}-{enemy_pos[0][1]}"
        path_to_enemy, temp_distance = self.get_shortest_path_to(env_state, enemy_pos_string)
        objective_path = self.get_closest_upgrade(env_state)



        if debug_print: print(f"MY POSITION: {agent_pos}")

        # SETUP END

        """
        New decision making:
        
        objective_set - boolean: True if the bot currently has an objective
        
        objective - keyword to tell the agent what plan it is currently executing
        (eg. ['move_to_safety_critical', 'set_trap', 'get_uppers', /
        'restrict_opponent', 'hunt_opponent', 'move_to_safety', 'NOP'])
        for objective in objectives:
            if condition is met:
                set objective
        
        calculate objective needs
        do objective
        
        objectives - list: [objective (string), ..., objective (string)] list of strings of possible objectives for the bot
        This list will be ordered so that when the bot iterates over the list it will stop when it finds an objective
        it can achieve
        
        #agent needs a strict reason to stay in danger_zone
        if agent in danger_zone:
            if agent needs to move to safety: (check if planned moves will leave you in danger_zone)
                move towards safety
            if agent has reason to stay in danger_zone: (check if current plan is still valid)
                follow plan
            else:
                move towards safety
        
        #Agent is not in danger_zone and looking to gain small advantages over its opponent...
        # ... or to come up with a plan to DESTROY his enemies
        if objective_set:
            if objective is still valid:
                follow plan
            else:
                objective_set = False
        objective = get_objective(objectives, env_state, *args)
        objective.get_next_step()
        """

        action = None

        # If agent in big-time danger
        if agent_pos in advanced_danger_zone:
            safe_tile = self.get_nearest_safe_tile(env_state)
            path_to_safety, cost = self.get_shortest_path_to(env_state, safe_tile)
            for i in range(len(path_to_safety) - 1):
                if advanced_danger_zone[path_to_safety[i]] <= i + 2:
                    if debug_print: print("moving to safety")
                    action = self.get_movement_direction(agent_pos, path_to_safety[1])
                    break

        # If agent can set a trap
        if action is None and len(self.enemy_pos) >= 3:
            if self.enemy_pos[-3] == self.enemy_pos[-2] == self.enemy_pos[-1]:
                kill_tile = self.find_closest_tile_that_explodes_on_tile(env_state, agent_pos, enemy_pos[0], solid_map)
                if kill_tile is not None:
                    setup_tile = self.find_closest_tile_that_explodes_on_tile(env_state, agent_pos, kill_tile,
                                                                              solid_map)
                    if agent_pos == kill_tile:
                        if debug_print: print("placing bomb on kill tile")
                        action = Bombots.BOMB
                    elif kill_tile in danger_zone:
                        if debug_print: print("moving to kill tile")
                        kill_path, kill_path_cost = self.get_shortest_path_to(env_state, kill_tile)
                        safety_cost = self.check_place_bomb(env_state, bomb_pos=[setup_tile, kill_tile],
                                                            agent_pos=kill_tile)
                        ticks = kill_path_cost + safety_cost + 2
                        if ticks < advanced_danger_zone[kill_tile]:
                            if debug_print: print("waiting on setup tile")
                            action = Bombots.NOP
                        else:
                            if debug_print: print("starting to moving towards kill tile")
                            action = self.get_movement_direction(agent_pos, kill_path[1])
                    elif agent_pos == setup_tile:
                        if agent_pos not in env_state['bomb_pos']:
                            if debug_print: print("placing bomb on setup tile")
                            action = Bombots.BOMB
                        else:
                            kill_path, kill_path_cost = self.get_shortest_path_to(env_state, kill_tile)
                            safety_cost = self.check_place_bomb(env_state, bomb_pos=[setup_tile, kill_tile], agent_pos=kill_tile)
                            ticks = kill_path_cost + safety_cost + 2
                            if ticks < advanced_danger_zone[agent_pos]:
                                if debug_print: print("waiting on setup tile")
                                action = Bombots.NOP
                            else:
                                if debug_print: print("starting to moving towards kill tile")
                                action = self.get_movement_direction(agent_pos, kill_path[1])
                    elif setup_tile is not None:
                        if debug_print: print("Moving towards setup tile")
                        set_up_path, cost = self.get_shortest_path_to(env_state, setup_tile)
                        action = self.get_movement_direction(agent_pos, set_up_path[1])

        if action is None and agent_pos in danger_zone:
            safe_path, _ = self.get_shortest_path_to(env_state, self.get_nearest_safe_tile(env_state))
            action = self.get_movement_direction(agent_pos, safe_path[1])

        if action is None:
            objective_path = self.get_closest_upgrade(env_state)
            if objective_path is None:
                objective_path, _ = self.get_shortest_path_to(env_state, enemy_pos[0])

            if len(objective_path) == 1:
                next_tile = objective_path[0]
            else:
                next_tile = objective_path[1]
            nx, ny = zip(next_tile)
            if next_tile in danger_zone:
                if debug_print: print("dont move")
                action = Bombots.NOP
            elif self.env.box_map[nx][ny] == 1:
                if debug_print: print("bomb")
                if self.check_place_bomb(env_state) != -1:
                    action = Bombots.BOMB
                else:
                    action = Bombots.NOP
            else:
                if env_state['self_pos'] == next_tile:
                    if self.check_place_bomb(env_state) != -1:
                        action = Bombots.BOMB
                    else:
                        action = Bombots.NOP
                else:
                    action = self.get_movement_direction(agent_pos, next_tile)
                    if debug_print: print("move towards enemy")

        # Sanity check:
        x, y = agent_pos[0], agent_pos[1]
        if action == Bombots.UP:
             y -= 1
        elif action == Bombots.DOWN:
            y += 1
        elif action == Bombots.LEFT:
            x -= 1
        elif action == Bombots.RIGHT:
            x += 1
        tile = (x, y)
        if tile in advanced_danger_zone:
            if advanced_danger_zone[tile] <= 1:
                action = Bombots.NOP

        # DECISION MAKING

        # if agent in dangerzone:
            # if enemy_path.next == safe:
                # move towards enemy
            # else:
                # move towards safe
        # else:
            # if path_to_upers is not None:
                # set objective to upers
            # else:
                # set objective to enemy
            # if objective_path.next in danger_zone:
                # NOP
            # elif objective_path.next == box:
                # if safe to place bomb:
                    # place bomb
            # else:
                # move towards objective

        if debug_print: print("MAKE DECISION")




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

def get_coord_string(tup):
    if len(tup) == 2:
        return f"{tup[0]}-{tup[1]}"

def get_coord_tuple(string):
    return (string.split("-")[0], string.split("-")[1])

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


"""

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
                path_to_nearest_safe_tile, temp_distance = self.get_shortest_path_to(env_state, next_safe_tile)
                move_to_safety = False
                advanced_danger_zone = self.get_advanced_danger_zone(env_state)
                if len(self.enemy_pos) >= 3:
                    if self.enemy_pos[-1] == self.enemy_pos[-2] == self.enemy_pos[-3]:
                        for i in range(len(path_to_nearest_safe_tile)):
                            if path_to_nearest_safe_tile[i] in advanced_danger_zone:
                                if advanced_danger_zone[path_to_nearest_safe_tile[i]] < i + 1:
                                    move_to_safety = True
                                    break
                if not move_to_safety:
                    kill_tile = self.find_closest_tile_that_explodes_on_tile(env_state, agent_pos, enemy_pos[0], solid_map)
                    if kill_tile is not None:
                        setup_tile = self.find_closest_tile_that_explodes_on_tile(env_state, agent_pos, kill_tile, solid_map)
                        if agent_pos == kill_tile:
                            action = Bombots.BOMB
                        # if agent has left setup_tile but not reached kill_tile
                        elif agent_pos == setup_tile:
                            if agent_pos not in env_state['bomb_pos']:
                                action = Bombots.BOMB
                            else:
                                kill_path, kill_path_cost = self.get_shortest_path_to(env_state, kill_tile)
                                safety_path, safety_path_cost = self.get_shortest_path_to(env_state, next_safe_tile, start=kill_tile)
                                ticks = kill_path_cost + safety_path_cost + 1
                                if ticks > advanced_danger_zone[agent_pos]:
                                    action = Bombots.NOP
                                else:
                                    action = self.get_movement_direction(agent_pos, kill_path[1])
                        elif kill_tile in danger_zone:
                            kill_path = self.get_shortest_path_to(env_state, kill_tile)
                            action = self.get_movement_direction(agent_pos, kill_path[1])
                        else:
                            set_up_path, cost = self.get_shortest_path_to(env_state, setup_tile)
                            action = self.get_movement_direction(agent_pos, set_up_path[1])
                    else:
                        next_tile_towards_safety = path_to_nearest_safe_tile[1]
                        action = self.get_movement_direction(agent_pos, next_tile_towards_safety)
                        pass
                else:
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
                    if self.check_place_bomb(env_state) != -1:
                        action = Bombots.BOMB
                    else:
                        action = Bombots.NOP
            else:
                if env_state['self_pos'] == next_tile and objective == 'bomb':
                    if self.check_place_bomb(env_state) != -1:
                        action = Bombots.BOMB
                    else:
                        action = Bombots.NOP
                else:
                    action = self.get_movement_direction(agent_pos, next_tile)
                    if debug_print: print("move towards enemy")
"""
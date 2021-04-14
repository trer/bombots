"""
2-input XOR example -- this is most likely the simplest possible example.
"""

from __future__ import print_function

import math
import os
import neat
import numpy as np

# 2-input XOR inputs and expected outputs.
xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
xor_outputs = [   (0.0,),     (1.0,),     (1.0,),     (0.0,)]



class NeatAgent:

    def __init__(self, env, config, genome):
        self.env = env
        self.config = config
        self.genome = genome
        self.net = neat.nn.FeedForwardNetwork.create(genome, config)
        self.states = []


    def eval_genomes(self, genomes, config):
        for genome_id, genome in genomes:
            genome.fitness = 4.0
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            for xi, xo in zip(xor_inputs, xor_outputs):
                output = net.activate(xi)
                genome.fitness -= (output[0] - xo[0]) ** 2

    def act(self, state, reward, done, info):
        self.states.append(state)
        state = np.copy(state)
        me_cords, opponent_cords, map = self.prep_state(state)
        s = np.concatenate((me_cords, opponent_cords))
        s = np.concatenate((s, map))
        output = self.net.activate(s)
        print(output)
        output = math.floor(output[0]*7)
        if output == 0:
            return self.env.NOP
        elif output == 1:
            return self.env.UP
        elif output == 2:
            return self.env.Down
        elif output == 0:
            return self.env.RIGHT
        elif output == 0:
            return self.env.LEFT
        else:
            output = self.env.BOMB
        return output

    def prep_state(self, state):
        me_cord = np.where(state[0] == 1)
        opponent_cord = np.where(state[1] == 1)
        """
        bomb_cords = np.where(state[2] == 1)
        fire_cords = np.where(state[3] == 1)
        crates_cords = np.where(state[4] == 1)
        solid_map_cords = np.where(state[5] == 1)
        """
        for i in range(2, len(state)):
            layer = state[i]
            state[i] = np.where(layer == 1, i-1, layer)

        map = np.maximum(state[2], state[3])
        map = np.maximum(map, state[4])
        map = np.maximum(map, state[5])
        map = map.flatten()
        return [me_cord[0][0], me_cord[1][0]], [opponent_cord[0][0], opponent_cord[1][0]], map

    def map_map(self, state):
        for i in range(1, len(state)):
            pass
        return state

    def run(self, config_file):
        # Load configuration.
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_file)

        # Create the population, which is the top-level object for a NEAT run.
        p = neat.Population(config)

        # Add a stdout reporter to show progress in the terminal.
        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        p.add_reporter(neat.Checkpointer(5))

        # Run for up to 300 generations.
        winner = p.run(self.eval_genomes, 300)

        # Display the winning genome.
        print('\nBest genome:\n{!s}'.format(winner))

        # Show output of the most fit genome against training data.
        print('\nOutput:')
        winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
        for xi, xo in zip(xor_inputs, xor_outputs):
            output = winner_net.activate(xi)
            print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))


        p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
        p.run(self.eval_genomes, 10)


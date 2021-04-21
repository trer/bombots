import numpy as np

from bombots.environment import Bombots
from templates.NEAT.NEAT import NeatAgent
from templates.agent_that_does_not_suicide import BeatNopAgent
from templates.NOP import NopAgent
import neat
import os

# For Travis
import sys
if '--novid' in sys.argv: 

    os.environ["SDL_VIDEODRIVER"] = "dummy"

env = Bombots(
    scale       = 64,                    # Size of game tiles (in pixels)
    framerate   = 10,                    # Frames per second, set this to 0 for unbounded framerate
    state_mode  = Bombots.STATE_TENSOR,  # So the state is returned as a tensor
    verbose     = False,                  # Useful printing during execution
    render_mode = Bombots.RENDER_GFX_RGB, # Change this to Bombots.NO_RENDER if you remove the render call
)
config_path = os.path.join('', 'templates/NEAT/config-feedforward')

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_path)

# Create the population, which is the top-level object for a NEAT run.
p = neat.Population(config)

# Add a stdout reporter to show progress in the terminal.
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(5))

def eval_genomes(genomes, config):
    player1_wins = 0
    player2_wins = 0
    last = None

    for genome_id, genome in genomes:
        agents = [NeatAgent(env, config, genome), NopAgent(env)]
        genome.fitness = 4.0
        states = env.reset()
        rewards = [0, 0]
        done = False
        info = {}
        while not done:
            states, rewards, done, info = env.step(
                [agents[i].act(states[i], rewards[i], done, info) for i in range(len(agents))])
            """
            bomb_cords = np.where(state[2] == 1)
            fire_cords = np.where(state[3] == 1)
            crates_cords = np.where(state[4] == 1)
            solid_map_cords = np.where(state[5] == 1)
            """
            if last is not None:
                same = True
                for i in range(len(states)):
                    if not np.array_equal(states[i], last[i]):
                        same = False
                        break
                if same:
                    pass
                    # if you want to punish standing still
                    genome.fitness -= 0.001
                else:
                    if genome.fitness < 10:
                        pass
                        # if you want to encourage moving
                        genome.fitness += 1
                    if not np.array_equal(states[0][4], last[0][4]):
                        # If you want to encourage bombing
                        print("It placed a bomb and it exploded something")
                        genome.fitness += 10
            last = states

            # env.render()  # Comment out this call to train faster

            if done:
                #print(done)
                #print(info)
                if player1_wins != info['player1']['wins']:
                    print("it fakking did it")
                    genome.fitness += 1000
                    player1_wins = info['player1']['wins']
                else:
                    genome.fitness -= 3


winner = p.run(eval_genomes, 1000)
agents = [NeatAgent(env, config, winner), BeatNopAgent(env)]
states = env.reset()

done = False
rewards = [0, 0]
info = {}
while True:
    states, rewards, done, info = env.step(
        [agents[i].act(states[i], rewards[i], done, info) for i in range(len(agents))])
    last = states
    env.render()  # Comment out this call to train faster
    if done:
        print(info)
        states = env.reset()

"""
node_names = {-1:'A', -2: 'B', 0:'A XOR B'}
    visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)
    """
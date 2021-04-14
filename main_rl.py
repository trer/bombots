import numpy as np

from bombots.environment import Bombots
from templates.agent_rl import RLAgent
from NEAT.NEAT import NeatAgent
from NOP import NopAgent
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
    verbose     = True,                  # Useful printing during execution
    render_mode = Bombots.RENDER_GFX_RGB # Change this to Bombots.NO_RENDER if you remove the render call
)
config_path = os.path.join('', 'NEAT/config-feedforward')

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
            if last is not None:
                same = True
                for i in range(len(states)):
                    if not np.array_equal(states[i], last[i]):
                        same = False
                        break
                if same == last:
                    genome.fitness -= 0.001
            last = states

            env.render()  # Comment out this call to train faster

            if done:
                print(done)
                if player1_wins != info['player1_wins']:
                    genome.fitness += 10
                    player1_wins = info['player1_wins']
                else:
                    genome.fitness -= 3

p.run(eval_genomes, 10)
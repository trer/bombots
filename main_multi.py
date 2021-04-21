import sys

import neat
import numpy as np

from bombots.environment import Bombots
import pygame as pg
from templates.NOP import NopAgent
from templates.agent_that_does_not_suicide import BeatNopAgent
from templates.NEAT.NEAT import NeatAgent

# For Travis
if '--novid' in sys.argv:
    import os

    os.environ["SDL_VIDEODRIVER"] = "dummy"


def eval_genomes(genomes):
    cell_scale = 16
    pg.init()

    # Initialize the master window
    screen = pg.display.set_mode((cell_scale * 12 * 4, cell_scale * 12 * 4))

    envs = [Bombots(
        # Size of game tiles (in pixels), of each individual game
        scale=cell_scale,
        # Frames per second, set this to 0 for unbounded framerate
        framerate=0,
        # Useful printing during execution
        verbose=True,
        # This is required to render multiple environments in the same window
        standalone=False,
        # For rule-based agents
        state_mode=Bombots.STATE_DICT
    ) for _ in range(4 * 4)]
    if '--test' not in sys.argv:

        player1_wins = 0
        player2_wins = 0
        last = None

        for genome_id, genomes in genomes:
            agents_a = [NeatAgent(env) for env in envs]
            agents_b = [NopAgent(env) for env in envs]
            genomes.fitness = 4.0
            states = [env.reset() for env in envs]
            rewards = [0, 0]
            done = False
            info = {}

        while True:
            for i, env in enumerate(envs):

                states[i], _, done, _ = env.step([agents_a[i].act(states[i][0]),
                                                  agents_b[i].act(states[i][1])])

                if last is not None:
                    same = True
                    for i in range(len(states)):
                        if not np.array_equal(states[i], last[i]):
                            same = False
                            break
                    if same:
                        pass
                        # if you want to punish standing still
                        genomes.fitness -= 0.001
                    else:
                        if genomes.fitness < 10:
                            pass
                            # if you want to encourage moving
                            genomes.fitness += 1
                        if not np.array_equal(states[0][4], last[0][4]):
                            # If you want to encourage bombing
                            print("It placed a bomb and it exploded something")
                            genomes.fitness += 10
                last = states
                env.render()

                # Paste each individual environment surface into the master window
                screen.blit(env.screen, ((i % 4) * (env.width + 1) * env.scale,
                                         (i // 4) * (env.height + 1) * env.scale))

                if done:
                    states[i] = env.reset()


def main():
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

    winner = p.run(eval_genomes, 10)

    env = Bombots(
        scale=64,  # Size of game tiles (in pixels)
        framerate=10,  # Frames per second, set this to 0 for unbounded framerate
        state_mode=Bombots.STATE_TENSOR,  # So the state is returned as a tensor
        verbose=False,  # Useful printing during execution
        render_mode=Bombots.RENDER_GFX_RGB,  # Change this to Bombots.NO_RENDER if you remove the render call
        seed=1
    )

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


if __name__ == '__main__':
    main()

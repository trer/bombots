import torch
import random
import numpy as np
from collections import deque

from templates.NOP import NopAgent
from templates.agent_that_does_not_suicide import BeatNopAgent
from bombots.environment import Bombots
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100000
BATCH_SIZE = 1000
LR = 0.001


class Agent:

    def __init__(self, env):
        self.danger_zone = {}
        self.training_agent = BeatNopAgent(env)
        self.env = env
        self.n_games = 2
        self.epsilon = 0.5  # randomness
        self.gamma = 0.99  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model = Linear_QNet(6, 6)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        self.long_state = deque([np.zeros((11, 11)) for ____ in range(35)], maxlen=35)    # popleft()
        self.model.apply(self.model.init_weights)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # popleft if MAX_MEMORY is reached

    def get_state(self, state):
        state = np.copy(state)
        for i in range(2, len(state)):
            layer = state[i]
            state[i] = np.where(layer == 1, i + 1, layer)

        for i in range(2):
            layer = state[i]
            state[i] = np.where(layer == 1, (len(state) + 1) * (i + 1), layer)

        map = np.add(state[0], state[1])
        map = np.add(map, state[2])
        map = np.add(map, state[3])
        map = np.add(map, state[4])
        map = np.add(map, state[5])
        return np.array(map, dtype=np.float32)

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        # print(len(states))
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def act(self, state):
        # random moves: tradeoff exploration / exploitation
        if self.n_games < 500:
            self.epsilon = 500 - self.n_games
        elif self.n_games < 10000:
            self.epsilon = 2
        final_move = [0, 0, 0, 0, 0, 0]
        if random.randint(0, 500) < self.epsilon:
            if self.n_games < 10000:
                move = random.randint(0, 5)
                final_move[move] = 1
            else:
                move = random.randint(0, 5)
                final_move[move] = 1
        else:
            #state = self.get_state(state)
            state1 = np.copy(state)
            state0 = torch.tensor(state1, dtype=torch.float).reshape(1, 35, 11, 11)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        if final_move[0] == 1:
            return self.env.BOMB
        elif final_move[1] == 1:
            return self.env.NOP
        elif final_move[2] == 1:
            return self.env.DOWN
        elif final_move[3] == 1:
            return self.env.RIGHT
        elif final_move[4] == 1:
            return self.env.LEFT
        elif final_move[5]:
            return self.env.UP
        raise Exception

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    env = Bombots(
        scale=64,  # Size of game tiles (in pixels)
        framerate=0,  # Frames per second, set this to 0 for unbounded framerate
        state_mode=Bombots.STATE_TENSOR,  # So the state is returned as a tensor
        verbose=False,  # Useful printing during execution
        render_mode=Bombots.RENDER_GFX_RGB,  # Change this to Bombots.NO_RENDER if you remove the render call RENDER_GFX_RGB
    )
    agents = [Agent(env), NopAgent(env)]
    agent = agents[0]
    nop = agents[1]
    states = env.reset()
    rewards = [0, 0]
    done = False
    info = {}
    last = {'wins': 0, 'boxes': 0, 'loss': 0}

    agent.long_state.append(agent.get_state(states[0]))
    state_old = np.array(agent.long_state, dtype=np.float32)#.reshape((1, 35, 11, 11))

    count_max = 100
    count = 0

    while True:
        # get move
        final_moves = [agents[i].act(state_old) for i in range(len(agents))]
        final_move = final_moves[0]


        # perform move and get new state
        state_new, rewards, done, info = env.step(
            [final_moves[i] for i in range(len(agents))])

        env.render()

        state_old1 = np.copy(state_old)
        #print("state_old", state_old1.shape)
        #state_old1 = np.reshape(state_old1, (1, 35, 6, 11, 11))
        agent.long_state.append(agent.get_state(state_new[0]))
        state_new1 = np.array(agent.long_state, dtype=np.float32)#.reshape((1, 35, 11, 11))
        #state_new1 = np.reshape(state_new1, (1, 35, 6, 11, 11))

        reward = rewards[0]


        # train short memory
        agent.train_short_memory(state_old1, final_move, reward, state_new1, done)

        # remember
        agent.remember(state_old1, final_move, reward, state_new1, done)

        state_old = np.array(agent.long_state)#.reshape((1, 35, 11, 11))

        if count < count_max:
            count += 1
        #if not done and info['player1']['boxes'] - last['boxes'] > 0:
        #    print("bombom not done")


        if done:
            # train long memory, plot result
            states = env.reset()

            agent.long_state = deque([np.zeros((11, 11)) for ____ in range(35)], maxlen=35)
            agent.long_state.append(agent.get_state(states[0]))
            state_old = np.array(agent.long_state)#.reshape((1, 35, 11, 11))
            agent.n_games += 1
            agent.train_long_memory()

            win = info['player1']['wins'] - last['wins']
            loss = info['player2']['wins'] - last['loss']
            boxes = info['player1']['boxes'] - last['boxes']
            last['wins'] = info['player1']['wins']
            last['boxes'] = info['player1']['boxes']
            last['loss'] = info['player2']['wins']

            score = 100 + 100*win + boxes + count/100 - 100*loss
            count = 0

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Wins', info['player1']['wins'], 'Boxes destroyed:', boxes)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()
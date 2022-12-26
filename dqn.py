from collections import deque
import math
import os
import random
from typing import Any
import gym
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from main import RandomPlayer

from quarto.objects import QuartoScape

env = QuartoScape()
print(env.observation_space)

# cartpile
env2 = gym.make('CartPole-v1')
print('CARTPOLE')
print(env2.observation_space.shape)

class DQNAgent:
    '''Play Quarto using a Deep Q-Network'''
    def __init__(self, env):
        self.env = env
        self.env.set_main_player(self)
        # main model updated every x steps
        self.model = self._build_model()
        # target model updated every y steps
        self.target_model = self._build_model()
        self.gamma = 0.95
        self.min_replay_size = 1000
        self.lr = 0.7
        self.epsilon = 0.7

    def get_all_actions(self):
        '''
        Return tuples from (0, 0, 0) to (3, 3, 15)
        Element 1 is position x
        Element 2 is position y
        Element 3 is piece chosen for next player
        '''
        tuples = []
        for i in range(0, 4):
            for j in range(0, 4):
                for k in range(0, 16):
                    tuples.append((i, j, k))
        return tuples

    def _build_model(self):
        '''
        Architecture of network:
        Input nodes are the state of the board
        Output nodes are the Q-values for each potential action (each output node is an action)
        An action is made up of (x, y, piece chosen for next player)
        There are 16 * 16 * 16 possible actions and the mapping is found in get_all_actions()
        '''
        model = Sequential()
        print('OBSERVATION SPACE')
        print(len(self.env.action_space.nvec))
        model.add(Dense(24, input_dim=self.env.observation_space.shape[0], activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(48, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(4 * 4 * 16, activation='softmax', kernel_initializer='he_uniform'))
        model.compile(loss='mse', metrics=['accuracy'], optimizer=Adam(lr=0.001))
        print(model.summary())
        return model

    def get_position(self, element, list):
        if element in list:
            return list.index(element)
        else:
            return -1

    def make_prediction(self, state, chosen_piece=None):
        '''Make a prediction using the network'''
        # prediction X is the position of the single 1 in the state
        pred_X = [self.get_position(i, list(state.flatten())) for i in range(0, 16)]
        pred_X.append(chosen_piece)
        # print('SHAPEEEE')
        # print(np.array([pred_X]).shape)
        # print(np.array([pred_X]))
        return self.model.predict(np.array([pred_X]))[0]

    def decay_lr(self, lr, decay_rate, decay_step):
        return lr * (1 / (1 + decay_rate * decay_step))

    def abbellire(self, state, action):
        '''
        Beautify the state for network input
        When in Italy, do as the Italians do
        '''
        X = [self.get_position(i, list(state.flatten())) for i in range(0, 16)]
        X.append(action[2])
        return np.array([X])

    def train(self, replay_memory, batch_size):
        '''Train the network'''
        if len(replay_memory) < self.min_replay_size:
            print('REPLAY MEMORY TOO SMALL')
            return

        print('TRAINING')
        batch_size = 64 * 2
        minibatch = random.sample(replay_memory, batch_size)
        # current_states = [transition[0] for transition in minibatch]
        current_states = np.array([self.abbellire(state, action) for state, action, reward, new_current_state, done in minibatch])
        # print('CURRENT STATES')
        # print(current_states.shape)
        current_qs = self.model.predict(current_states)
        # print('CURRENT QS')
        # print(current_qs.shape)
        # new_current_states = np.array([transition[3] for transition in minibatch])
        new_current_states = np.array([self.abbellire(new_current_state, action) for state, action, reward, new_current_state, done in minibatch])
        future_qs = self.target_model.predict(new_current_states)

        X = []
        Y = []
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs[index])
                new_q = reward + self.gamma * max_future_q
            else:
                new_q = reward

            current_qs[index][0][action[0] + action[1] * 4 + action[2] * 16] = new_q

            X.append(self.abbellire(current_state, action))
            Y.append(current_qs[index])

        X = np.array(X).reshape(batch_size, 17)
        Y = np.array(Y).reshape(batch_size, 4 * 4 * 16)
        print(X.shape)
        self.model.fit(X, Y, batch_size=batch_size, verbose=2, shuffle=False, epochs=5)

    def choose_piece(self, state: Any):
        '''Choose piece for the next guy to play'''
        pred = self.make_prediction(state)
        return pred[2]

    def place_piece(self, state: Any):
        '''Choose position to move piece to based on the current state'''
        pred = self.make_prediction(state)
        coordinates = pred[0], pred[1]
        return (coordinates % 4, coordinates // 4)

    def zero_out_invalid_actions(self, current_piece, prediction):
        '''Zero out invalid moves'''
        # zero out invalid moves
        all_actions = self.get_all_actions()
        for i in range(len(prediction)):
            action = all_actions[i]
            if not self.env.game.check_if_move_valid(current_piece, action[0], action[1], action[2]):
                prediction[i] = -math.inf

        return prediction


    def run(self):
        '''Run training of agent for x episodes'''
        # ensure both model and target model have same set of weights at the start
        self.target_model.set_weights(self.model.get_weights())

        replay_memory = deque(maxlen=5000)
        state = self.env.reset()
        # number of episodes to train for
        num_episodes = 100

        steps_to_update_target_model = 0

        for episode in range(num_episodes):
            total_training_reward = 0
            print(f'Episode: {episode}')
            state = self.env.reset()
            done = False
            # initialise chosen piece with a random piece
            # in reality, the opponent will choose a piece for you
            chosen_piece = random.randint(1, 15)
            while not done:
                print('STATE')
                steps_to_update_target_model += 1

                if random.random() < self.epsilon:
                    action = self.env.action_space.sample()
                    while not self.env.game.check_if_move_valid(chosen_piece, action[0], action[1], action[2]):
                        print(f'INVALID ACTION: {action}')
                        print(f'CHOSEN PIECE: {chosen_piece}')
                        action = self.env.action_space.sample()
                    print('RANDOM ACTION')
                    print(action)
                else:
                    prediction = self.make_prediction(state, chosen_piece)
                    print('p1')
                    prediction = self.zero_out_invalid_actions(chosen_piece, prediction)
                    print('p2')
                    action = np.argmax(prediction)
                    print('p3')
                    # action = np.argmax(self.make_prediction(state, chosen_piece))
                    # get action at index of action
                    action = self.get_all_actions()[action]
                    print('PREDICTED ACTION')
                    print(action)

                new_state, reward, done, _ = self.env.step(action, chosen_piece)
                replay_memory.append((state, action, reward, new_state, done))

                if done:
                    print('GAME OVER')

                if steps_to_update_target_model % 4 == 0 or done:
                    print('Training')
                    self.train(replay_memory, 64)

                state = new_state
                total_training_reward += reward

                if done:
                    print(f'Total reward: {total_training_reward} at episode {episode} after {steps_to_update_target_model} steps')
                    total_training_reward += 1

                    if steps_to_update_target_model >= 100:
                        self.target_model.set_weights(self.model.get_weights())
                        steps_to_update_target_model = 0
                    break

                chosen_piece = action[2]

            self.epsilon = self.decay_lr(self.lr, 0.0001, episode)
        self.env.close()
        self.model.save('model.h5')

agent = DQNAgent(env)
agent.run()
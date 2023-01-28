import argparse
from collections import deque
import logging
import math
import os
import random
from typing import Any
import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import HeUniform

from quarto.objects import Player, Quarto, QuartoScape

env = QuartoScape()
# print(env.observation_space)

# cartpile
env2 = gym.make('CartPole-v1')
# print('CARTPOLE')
# print(env2.observation_space.shape)

class RandomPlayer(Player):
    """Random player"""

    def __init__(self, quarto: Quarto):
        super().__init__(quarto)

    def choose_piece(self, state=None):
        return random.randint(0, 15)

    def place_piece(self, state=None, piece_to_be_placed: int = None):
        return random.randint(0, 3), random.randint(0, 3)

def test(agent):
    dq_wins = 0
    for round in range(100):
        game = Quarto()
        agent.set_game(game)
        game.set_players((RandomPlayer(game), agent))
        winner = game.run()
        if winner == 1:
            dq_wins += 1
        # logging.warning(f"main: Winner: player {winner}")
    logging.warning(f"main: DQ wins: {dq_wins}")

class DQNAgent:
    '''Play Quarto using a Deep Q-Network'''
    def __init__(self, env=env, game=None):
        self.env = env
        # self.env.set_main_player(self)
        # main model updated every x steps
        self.model = self.build_conv_model()
        # target model updated every y steps
        self.target_model = self.build_conv_model()
        self.gamma = 0.618
        self.min_replay_size = 500
        self.lr = 0.7
        self.epsilon = 0.5
        if game is not None:
            self.env.game = game

        # if os.path.exists('model.h5'):
        #     # print('Loading model')
        #     self.model = tf.keras.models.load_model('model.h5')

    def set_game(self, game):
        self.env.game = game

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
        # print('OBSERVATION SPACE')
        # print(len(self.env.action_space.nvec))
        initializer = HeUniform()
        model.add(Dense(12, input_dim=self.env.observation_space.shape[0], activation='relu', kernel_initializer=initializer))
        model.add(Dense(24, activation='relu', kernel_initializer=initializer))
        model.add(Dense(48, activation='relu', kernel_initializer=initializer))
        model.add(Dense(96, activation='relu', kernel_initializer=initializer))
        # model.add(Dense(192, activation='relu', kernel_initializer=initializer))
        # model.add(Dense(384, activation='relu', kernel_initializer=initializer))
        # model.add(Dense(768, activation='relu', kernel_initializer=initializer))
        model.add(Dense(4 * 4 * 16, activation='linear', kernel_initializer=initializer))
        model.compile(loss=tf.keras.losses.Huber(), metrics=['mae', 'mse'], optimizer=Adam(lr=0.001))
        # print(model.summary())
        return model

    def build_conv_model(self):
        '''
        Two inputs:
        4 x 4 x 4 board
        1 x 4 piece
        1 conv layer
        1 dense layer
        '''
        input1 = tf.keras.Input(shape=(4, 4, 4), name='x1')
        input2 = tf.keras.Input(shape=(1, 4), name='x2')
        x = Conv2D(16, (2, 2), activation='relu')(input1)
        x = MaxPooling2D((2, 2))(x)
        x = Flatten()(x)
        x2 = tf.keras.layers.Reshape((1, 4))(input2)
        x2 = Dense(16, activation='relu')(x2)
        x2 = tf.squeeze(x2, axis=1)
        x = tf.keras.layers.concatenate([x, x2])
        x = Dense(64, activation='relu')(x)
        x = Dense(4 * 4 * 16, activation='linear')(x)
        model = tf.keras.Model(inputs={'x1': input1, 'x2': input2}, outputs=x)
        model.compile(loss=tf.keras.losses.Huber(), metrics=['mae', 'mse'], optimizer=Adam(lr=0.001))
        return model

    def get_position(self, element, list):
        if element in list:
            return list.index(element)
        else:
            return -1

    def make_prediction(self, state, chosen_piece=None):
        '''Make a prediction using the network'''
        # prediction X is the position of the single 1 in the state
        x = self.abbellire(state, chosen_piece)
        x = {'x1': np.array([x[0]]), 'x2': np.array([x[1]])}
        # pred_X.append(chosen_piece)
        # # print(np.array([pred_X]))
        return self.model.predict(x)

    def decay_lr(self, lr, decay_rate, decay_step):
        return lr * (1 / (1 + decay_rate * decay_step))

    def abbellire(self, state, chosen_piece):
        new_rep = np.zeros((4, 4, 4))
        for i in range(0, 4):
            for j in range(0, 4):
                piece = state[i][j]
                if piece == -1:
                    continue
                piece = self.env.game.get_pieces()[piece]
                high = piece.HIGH
                coloured = piece.COLOURED
                solid = piece.SOLID
                square = piece.SQUARE
                new_rep[i, j, 0] = high
                new_rep[i, j, 1] = coloured
                new_rep[i, j, 2] = solid
                new_rep[i, j, 3] = square
        x1 = new_rep
        x2 = np.zeros((1, 4))
        piece = self.env.game.get_pieces()[chosen_piece]
        x2[0, 0] = piece.HIGH
        x2[0, 1] = piece.COLOURED
        x2[0, 2] = piece.SOLID
        x2[0, 3] = piece.SQUARE

        return x1, x2

    # def create_X(self, state, chosen_piece):
    #     X = [self.a(i, list(state.flatten())) for i in range(0, 16)]
    #     X.append(chosen_piece)
    #     return np.array([X])

    def train(self, replay_memory, batch_size):
        '''Train the network'''
        if len(replay_memory) < self.min_replay_size:
            # print('REPLAY MEMORY TOO SMALL')
            print("Replay memory ", len(replay_memory))
            return

        # print('TRAINING')
        batch_size = 64 * 2
        minibatch = random.sample(replay_memory, batch_size)
        # current_states = [transition[0] for transition in minibatch]
        # state + chosen_piece for you -> action (contains chosen_piece for next player)
        current_states = [self.abbellire(state, chosen_piece) for state, chosen_piece, action, reward, new_current_state, done in minibatch]
        print({'x1': [s[0][0] for s in current_states], 'x2': [s[1][0] for s in current_states]})
        current_qs = self.model.predict({'x1': [s[0][0] for s in current_states], 'x2': [s[1][0] for s in current_states]})
        # new current state + chosen_piece for next player -> action (contains chosen_piece for next player)
        new_current_states = [self.abbellire(new_current_state, action[2]) for state, chosen_piece, action, reward, new_current_state, done in minibatch]
        future_qs = self.target_model.predict({'x1': [s[0] for s in new_current_states], 'x2': [s[1] for s in new_current_states]})
        # exclude invalid moves from calculation
        # future_qs = [self.nan_out_invalid_actions(batch[2][2], future_q) for batch, future_q in zip(minibatch, future_qs)]

        X = []
        Y = []
        for index, (current_state, chosen_piece, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                # max_future_q = np.max(future_qs[index])
                # new_q = reward + self.gamma * max_future_q
                max_future_q = reward + self.gamma * np.max(future_qs[index])
            else:
                # max_future_q = reward
                max_future_q = reward

            # 0 2 5
            # 0 + 2 * 4 + 5 * 16 = 85

            current_qs[index][action[0] + action[1] * 4 + action[2] * 16] = (1 - self.lr) * current_qs[index][action[0] + action[1] * 4 + action[2] * 16] + self.lr * max_future_q

            beautified = self.abbellire(current_state, chosen_piece)
            X.append({'x1': beautified[0], 'x2': beautified[1]})
            Y.append(current_qs[index])

        # X = np.array(X).reshape(batch_size, 17)
        # X = np.array(X).reshape(batch_size, 17)
        X = np.array({'x1': [s[0] for s in X], 'x2': [s[1] for s in X]})
        Y = np.array(Y).reshape(batch_size, 4 * 4 * 16)
        self.model.fit(X, Y, batch_size=batch_size, verbose=2, shuffle=True)

    def choose_piece(self, state: Any, piece_chosen_for_you: int):
        '''Choose piece for the next guy to play'''
        self.env.game.__board = state
        pred = self.make_prediction(state, piece_chosen_for_you)
        pred = self.nan_out_invalid_actions(piece_chosen_for_you, pred)
        # print(f'Number of valid moves: {len([i for i in pred if i != np.nan])}')
        best_action = np.nanargmax(pred)
        best_action = self.get_all_actions()[best_action]
        return best_action[2]

    def place_piece(self, state: Any, piece_chosen_for_you: int):
        '''Choose position to move piece to based on the current state'''
        # print(f'PIECE CHOSEN FOR YOU: {piece_chosen_for_you}')
        self.env.game.__board = state
        pred = self.make_prediction(state, piece_chosen_for_you)
        pred = self.nan_out_invalid_actions(piece_chosen_for_you, pred)
        # print(f'Number of valid moves: {len([i for i in pred if i != np.nan])}')
        best_action = np.nanargmax(pred)
        best_action = self.get_all_actions()[best_action]
        return best_action[0], best_action[1]

    def nan_out_invalid_actions(self, current_piece, prediction):
        '''Zero out invalid moves'''
        # zero out invalid moves
        all_actions = self.get_all_actions()
        for i in range(len(prediction)):
            action = all_actions[i]
            # print(action)
            # print(current_piece)
            if not self.env.game.check_if_move_valid(current_piece, action[0], action[1], action[2]):
                prediction[i] = np.nan

        return prediction

    def run(self):
        '''Run training of agent for x episodes'''
        # ensure both model and target model have same set of weights at the start
        self.target_model.set_weights(self.model.get_weights())

        replay_memory = deque(maxlen=5000)
        state = self.env.reset()
        # number of episodes to train for
        num_episodes = 2000

        steps_to_update_target_model = 0

        for episode in range(num_episodes):
            total_training_reward = 0
            print(f'Episode: {episode}')
            state = self.env.reset()
            done = False
            # initialise chosen piece with a random piece
            # in reality, the opponent will choose a piece for you
            chosen_piece = random.randint(0, 15)
            while not done:
                # print('STATE')
                steps_to_update_target_model += 1

                if random.random() < self.epsilon:
                    action = self.env.action_space.sample()
                    while not self.env.game.check_if_move_valid(chosen_piece, action[0], action[1], action[2]):
                        # print(f'INVALID ACTION: {action}')
                        # print(f'CHOSEN PIECE: {chosen_piece}')
                        action = self.env.action_space.sample()
                    # print('RANDOM ACTION')
                    # print(action)
                else:
                    prediction = self.make_prediction(state, chosen_piece)
                    prediction = self.nan_out_invalid_actions(chosen_piece, prediction)
                    if np.all(np.isnan(prediction)):
                        # print('ALL NAN')
                        action = self.env.action_space.sample()
                        while not self.env.game.check_if_move_valid(chosen_piece, action[0], action[1], action[2]):
                            # print(f'INVALID ACTION: {action}')
                            # print(f'CHOSEN PIECE: {chosen_piece}')
                            action = self.env.action_space.sample()
                    else:
                        action = np.nanargmax(prediction)
                        # action = np.argmax(self.make_prediction(state, chosen_piece))
                        # get action at index of action
                        action = self.get_all_actions()[action]
                    # print('PREDICTED ACTION')
                    # print(action)

                new_state, reward, done, _ = self.env.step(action, chosen_piece)
                replay_memory.append((state, chosen_piece, action, reward, new_state, done))

                # if done:
                #     print('GAME OVER')

                if steps_to_update_target_model % 4 == 0 or done:
                    # print('Training')
                    self.train(replay_memory, 64)

                state = new_state
                total_training_reward += reward

                if done:
                    # print(f'Total reward: {total_training_reward} at episode {episode} after {steps_to_update_target_model} steps')
                    total_training_reward += 1

                    if steps_to_update_target_model >= 100:
                        self.target_model.set_weights(self.model.get_weights())
                        steps_to_update_target_model = 0
                    break

                chosen_piece = action[2]

            if episode % 10 == 0:
                test(self)

            self.lr = self.decay_lr(self.lr, 0.0001, episode)
        self.env.close()
        self.model.save('model.h5')

agent = DQNAgent(env)
agent.run()

# def main():
#     dq_wins = 0
#     for round in range(100):
#         game = Quarto()
#         game.set_players((RandomPlayer(game), DQNAgent(game=game)))
#         winner = game.run()
#         if winner == 1:
#             dq_wins += 1
#         logging.warning(f"main: Winner: player {winner}")
#     logging.warning(f"main: DQ wins: {dq_wins}")
#     # game = Quarto()
#     # game.set_players((RandomPlayer(game), DQNAgent(game=game)))
#     # dq_wins = 0
#     # # for round in range(100):
#     # winner = game.run()
#     # if winner == 1:
#     #     dq_wins += 1
#     # logging.warning(f"main: Winner: player {winner}")
#     # logging.warning(f"main: DQ wins: {dq_wins}")


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-v', '--verbose', action='count', default=0, help='increase log verbosity')
#     parser.add_argument('-d',
#                         '--debug',
#                         action='store_const',
#                         dest='verbose',
#                         const=2,
#                         help='log debug messages (same as -vv)')
#     args = parser.parse_args()

#     if args.verbose == 0:
#         logging.getLogger().setLevel(level=logging.WARNING)
#     elif args.verbose == 1:
#         logging.getLogger().setLevel(level=logging.INFO)
#     elif args.verbose == 2:
#         logging.getLogger().setLevel(level=logging.DEBUG)

#     main()
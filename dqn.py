import argparse
from collections import deque
import logging
import math
import os
import random
from typing import Any
import gym
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import HeUniform

from quarto.objects import Player, Quarto, QuartoScape

env = QuartoScape()
print(env.observation_space)

# cartpile
env2 = gym.make('CartPole-v1')
print('CARTPOLE')
print(env2.observation_space.shape)

class DQNAgent:
    '''Play Quarto using a Deep Q-Network'''
    def __init__(self, env=env, game=None):
        self.env = env
        # self.env.set_main_player(self)
        # main model updated every x steps
        self.model = self._build_model()
        # target model updated every y steps
        self.target_model = self._build_model()
        self.gamma = 0.95
        self.min_replay_size = 500
        self.lr = 0.7
        self.epsilon = 0.5
        if game is not None:
            self.env.game = game

        if os.path.exists('model.h5'):
            self.model.load_weights('model.h5')

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
        initializer = HeUniform()
        model.add(Dense(48, input_dim=self.env.observation_space.shape[0], activation='relu', kernel_initializer=initializer))
        model.add(Dense(24, activation='relu', kernel_initializer=initializer))
        model.add(Dense(12, activation='relu', kernel_initializer=initializer))
        model.add(Dense(4 * 4 * 16, activation='softmax', kernel_initializer=initializer))
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

    def create_X(self, state, chosen_piece):
        X = [self.get_position(i, list(state.flatten())) for i in range(0, 16)]
        X.append(chosen_piece)
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
                # max_future_q = np.max(future_qs[index])
                # new_q = reward + self.gamma * max_future_q
                max_future_q = reward + self.gamma * np.max(future_qs[index])
            else:
                # max_future_q = reward
                max_future_q = reward

            current_qs[index][0][action[0] + action[1] * 4 + action[2] * 16] = (1 - self.lr) * current_qs[index][0][action[0] + action[1] * 4 + action[2] * 16] + self.lr * max_future_q

            X.append(self.abbellire(current_state, action))
            Y.append(current_qs[index])

        # X = np.array(X).reshape(batch_size, 17)
        X = np.array(X)
        Y = np.array(Y).reshape(batch_size, 4 * 4 * 16)
        print(X.shape)
        self.model.fit(X, Y, batch_size=batch_size, verbose=2, shuffle=False, epochs=1)

    def choose_piece(self, state: Any, piece_chosen_for_you: int):
        '''Choose piece for the next guy to play'''
        self.env.game.__board = state
        pred = self.make_prediction(state, piece_chosen_for_you)
        pred = self.zero_out_invalid_actions(piece_chosen_for_you, pred)
        print(f'Number of valid moves: {len([i for i in pred if i != -math.inf])}')
        best_action = np.argmax(pred)
        best_action = self.get_all_actions()[best_action]
        return best_action[2]

    def place_piece(self, state: Any, piece_chosen_for_you: int):
        '''Choose position to move piece to based on the current state'''
        print(f'PIECE CHOSEN FOR YOU: {piece_chosen_for_you}')
        self.env.game.__board = state
        pred = self.make_prediction(state, piece_chosen_for_you)
        pred = self.zero_out_invalid_actions(piece_chosen_for_you, pred)
        print(f'Number of valid moves: {len([i for i in pred if i != -math.inf])}')
        best_action = np.argmax(pred)
        best_action = self.get_all_actions()[best_action]
        return best_action[0], best_action[1]

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
        num_episodes = 2000

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
                    prediction = self.zero_out_invalid_actions(chosen_piece, prediction)
                    action = np.argmax(prediction)
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

            self.lr = self.decay_lr(self.lr, 0.0001, episode)
        self.env.close()
        self.model.save('model.h5')

class RandomPlayer(Player):
    """Random player"""

    def __init__(self, quarto: Quarto):
        super().__init__(quarto)

    def choose_piece(self, state=None, piece_to_be_chosen: int = None):
        return random.randint(0, 15)

    def place_piece(self, state=None, piece_to_be_placed: int = None):
        return random.randint(0, 3), random.randint(0, 3)

# agent = DQNAgent(env)
# agent.run()

def main():
    dq_wins = 0
    for round in range(100):
        game = Quarto()
        game.set_players((RandomPlayer(game), DQNAgent(game=game)))
        winner = game.run()
        if winner == 1:
            dq_wins += 1
        logging.warning(f"main: Winner: player {winner}")
    logging.warning(f"main: DQ wins: {dq_wins}")
    # game = Quarto()
    # game.set_players((RandomPlayer(game), DQNAgent(game=game)))
    # dq_wins = 0
    # # for round in range(100):
    # winner = game.run()
    # if winner == 1:
    #     dq_wins += 1
    # logging.warning(f"main: Winner: player {winner}")
    # logging.warning(f"main: DQ wins: {dq_wins}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='count', default=0, help='increase log verbosity')
    parser.add_argument('-d',
                        '--debug',
                        action='store_const',
                        dest='verbose',
                        const=2,
                        help='log debug messages (same as -vv)')
    args = parser.parse_args()

    if args.verbose == 0:
        logging.getLogger().setLevel(level=logging.WARNING)
    elif args.verbose == 1:
        logging.getLogger().setLevel(level=logging.INFO)
    elif args.verbose == 2:
        logging.getLogger().setLevel(level=logging.DEBUG)

    main()
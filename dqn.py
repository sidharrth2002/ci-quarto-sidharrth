from collections import deque
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
        print(self.env.game.get_players())
        # main model updated every x steps
        self.model = self._build_model()
        # target model updated every y steps
        self.target_model = self._build_model()
        self.gamma = 0.95
        self.min_replay_size = 1000
        self.lr = 0.7
        self.epsilon = 0.7

    def _build_model(self):
        '''
        Architecture of network:
        Input nodes are the state of the board
        Output nodes are the Q-values for each action (each output node is an action)
        '''
        model = Sequential()
        print(self.env.observation_space.shape)
        print('SAMPLING ACTION')
        print(self.env.action_space.sample())
        model.add(Dense(24, input_dim=self.env.observation_space.shape[0] * self.env.observation_space.shape[1], activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(48, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.env.action_space.n, activation='linear', kernel_initializer='he_uniform'))
        model.compile(loss='mse', metrics=['accuracy'], optimizer=Adam(lr=0.001))
        print(model.summary())
        return model

    def make_prediction(self, state):
        '''Make a prediction using the network'''
        print('PREDICTION')
        print('part 1')
        print(self.model.predict(state.reshape([1, state.shape[0] * state.shape[1]])))
        return self.model.predict(state.reshape([1, state.shape[0] * state.shape[1]]))[0]

    def decay_lr(self, lr, decay_rate, decay_step):
        return lr * (1 / (1 + decay_rate * decay_step))

    def train(self, replay_memory, batch_size):
        '''Train the network'''
        if len(replay_memory) < self.min_replay_size:
            return

        batch_size = 64 * 2
        minibatch = random.sample(replay_memory, batch_size)
        current_states = np.array([transition[0] for transition in minibatch])
        current_qs = self.model.predict(current_states)
        print('CURRENT QS')
        print(current_qs)
        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs = self.target_model.predict(new_current_states)

        X = []
        Y = []
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs[index])
                new_q = reward + self.gamma * max_future_q
            else:
                new_q = reward

            current_qs[index][action] = (1 - self.lr) * current_qs[index][action] + self.lr * new_q

            X.append(current_state)
            Y.append(current_qs[index])

        self.model.fit(np.array(X), np.array(Y), batch_size=batch_size, verbose=0, shuffle=False)

    def choose_piece(self, state: Any) -> int:
        '''Choose piece to play based on the current state'''
        pred = self.make_prediction(state)
        print('PREDICTION')
        print(pred)
        return round(pred[0])

    def place_piece(self, state: Any) -> int:
        '''Choose position to move piece to based on the current state'''
        pred = self.make_prediction(state)
        return round(pred[1]), round(pred[2])

    def run(self):
        '''Run training of agent for x episodes'''
        # ensure both model and target model have same set of weights at the start
        self.target_model.set_weights(self.model.get_weights())

        replay_memory = deque(maxlen=5000)

        # number of episodes to train for
        num_episodes = 100

        steps_to_update_target_model = 0

        for episode in range(num_episodes):
            total_training_reward = 0
            print(f'Episode: {episode}')
            state = self.env.reset()
            done = False
            episode_reward = 0
            while not done:
                steps_to_update_target_model += 1
                # self.env.render()
                if random.random() <= self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    print('Making prediction')
                    print(state)
                    action = self.make_prediction(state)
                    action = np.argmax(action)
                new_state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                replay_memory.append((state, action, reward, new_state, done))
                state = new_state

                if steps_to_update_target_model % 4 == 0 or done:
                    self.train(replay_memory, batch_size=64)

                observation = new_state
                total_training_reward += reward

                if done:
                    print(f'Episode: {episode}, Reward: {episode_reward}')
                    total_training_reward += 1

                    if steps_to_update_target_model >= 100:
                        self.target_model.set_weights(self.model.get_weights())
                        print('Updated target model')
                        steps_to_update_target_model = 0
                    break
            epsilon = self.decay_lr(epsilon, 0.001, episode)

        self.env.close()

agent = DQNAgent(env)
agent.run()
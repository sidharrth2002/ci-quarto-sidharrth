from collections import deque
import random
from typing import Any
import gym
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from quarto.objects import QuartoScape

env = QuartoScape()

class DQNAgent:
    '''Play Quarto using a Deep Q-Network'''
    def __init__(self, env):
        self.env = env
        # main model updated every x steps
        self.model = self._build_model()
        # target model updated every y steps
        self.target_model = self._build_model()
        self.gamma = 0.95
        self.min_replay_size = 1000
        self.lr = 0.7

    def _build_model(self):
        '''
        Architecture of network:
        Input nodes are the state of the board
        Output nodes are the Q-values for each action (each output node is an action)
        '''
        model = Sequential()
        model.add(Dense(24, input_dim=self.env.observation_space.shape[0], activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(48, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.env.action_space.n, activation='linear', kernel_initializer='he_uniform'))
        model.compile(loss='mse', metrics=['accuracy'], optimizer=Adam(lr=0.001))
        return model

    def make_prediction(self, state):
        '''Make a prediction using the network'''
        return self.model.predict(state.reshape([1, state.shape[0]]))[0]

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

    def __call__(self, *args: Any, **kwds: Any):
        '''Run training of agent for x episodes'''
        # ensure both model and target model have same set of weights at the start
        self.target_model.set_weights(self.model.get_weights())

        replay_memory = deque(maxlen=50000)

        # number of episodes to train for
        num_episodes = 1000

        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            episode_reward = 0
            while not done:
                # self.env.render()
                if random.random() <= self.epsilon:
                    action = self.env.action_space.sample()
                else:
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
from collections import defaultdict
from copy import deepcopy
import json
import logging
import math
import random

from mcts import MonteCarloTreeSearch, MonteCarloTreeSearchDecoder, decode_tree
from quarto.objects import Quarto
from lib.players import RandomPlayer
from lib.isomorphic import BoardTransforms

logging.basicConfig(level=logging.INFO)


class QLearningPlayer:
    def __init__(self, board: Quarto = Quarto(), epsilon=0.1, alpha=0.5, gamma=0.9, tree: MonteCarloTreeSearch = None):
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.board = board
        self.MAX_PIECES = 16
        self.BOARD_SIDE = 4
        self.Q = defaultdict(int)

        if tree is not None:
            # load the pre-initalised tree
            self.tree = tree
            self.tree.set_board(board)

        else:
            # load new tree
            self.tree = MonteCarloTreeSearch(board=board)

    def reduce_normal_form(self, state: Quarto):
        '''
        Reduce the Quarto board to normal form (i.e. the board is symmetric)
        '''
        # NOT IMPLEMENTED for now, just return the board
        return state

    def hash_state_action(self, state: Quarto, action):
        # reduce to normal form before saving to Q table
        return state.board_to_string() + '||' + str(state.get_selected_piece()) + '||' + str(action)

    def get_Q(self, state, action):
        # check possible transforms first (really really slow)
        for key, val in self.Q.items():
            if BoardTransforms.compare_boards(state.state_as_array(), state.string_to_board(key.split('||')[0])):
                return val

        if self.hash_state_action(state, action) not in self.Q:
            return random.uniform(1.0, 0.01)

        return self.Q[self.hash_state_action(state, action)]

    def get_Q_for_state(self, state):
        if self.hash_state_action(state, None) not in self.Q:
            return None
        return [i for i in self.Q if i.startswith(str(state))]

    def set_Q(self, state, action, value):
        self.Q[self.hash_state_action(state, action)] = value

    def get_possible_actions(self, state: Quarto):
        actions = []
        for i in range(self.BOARD_SIDE):
            for j in range(self.BOARD_SIDE):
                for piece in range(self.MAX_PIECES):
                    if state.check_if_move_valid(self.board.get_selected_piece(), i, j, piece):
                        actions.append((i, j, piece))

        return actions

    def get_max_Q(self, state):
        max_Q = -math.inf
        for action in self.get_possible_actions(state):
            if self.get_Q(state, action) is not None:
                max_Q = max(max_Q, self.get_Q(state, action))
        return max_Q

    def get_action(self, state, mode='training'):
        '''
        If state, action pair not in Q, go to Monte Carlo Tree Search to find best action
        '''
        if mode == 'training':
            # TESTING mode (primarily use the Q table)
            if random.random() < self.epsilon:
                best_action = self.tree.place_piece()
                print('Best action: ', best_action)
                print(best_action)
                return best_action
                # return random.choice(self.get_possible_actions(state))
            else:
                expected_score = 0
                best_action = None
                for action in self.get_possible_actions(state):
                    if self.get_Q(state, action) is not None and expected_score < self.get_Q(state, action):
                        expected_score = self.get_Q(state, action)
                        best_action = action
                # go to Monte Carlo Tree Search if no suitable action found in Q table
                if best_action is None or expected_score == 0:
                    logging.info(
                        'No suitable action found in Q table, going to Monte Carlo Tree Search')
                    best_action = self.tree.place_piece()
                return best_action
        else:
            # TESTING mode (primarily use Q table)
            action = self.tree.place_piece()
            return action

    def update_Q(self, state, action, reward, next_state):
        self.set_Q(state, action, self.get_Q(state, action) + self.alpha *
                   (reward + self.gamma * self.get_max_Q(next_state) - self.get_Q(state, action)))

    def train(self, iterations=100):
        '''
        The basic idea behind MCTS-QL is to use MCTS to identify promising actions, and then use Q-learning to update the Q-values of those actions. The process can be described as follows:

        1. Use the Q-function to initialize the value of each state-action pair, Q(s, a) = 0.

        2. Use MCTS to select the next action to take by selecting the action with the highest value. The action value is the sum of the Q-value and a confidence value, computed as follows:
        Q'(s,a) = Q(s,a) + Cp * sqrt(ln(N(s))/N(a,s))
        where Cp is a constant, N(s) is the number of times the state s has been visited and N(a,s) is the number of times the action a has been taken from the state s.

        3. Take the selected action and observe the resulting state and reward.

        4. Use Q-learning to update the Q-value for the state-action pair that led to the new state using the following update rule:
        Q(s, a) = Q(s, a) + α * (r + γ * max(Q(s', a')) - Q(s, a))
        where s' is the new state, a' is the next action, r is the reward, γ is the discount factor and α is the learning rate.

        5. Repeat the process for multiple episodes.
        '''
        # 1. Use the Q-function to initialize the value of each state-action pair, Q(s, a) = 0.
        # automatically done through defaultdict

        # Choose an action using MCTS
        for i in range(iterations):
            board = Quarto()
            self.board = board
            random_player = RandomPlayer(board)
            self.tree.set_board(board)
            self.current_state = board
            self.previous_state = None
            self.previous_action = None
            player = 0
            selected_piece = random_player.choose_piece()
            self.current_state.set_selected_piece(selected_piece)
            while True:
                reward = 0
                if player == 0:
                    # QL-MCTS moves here
                    self.previous_state = deepcopy(self.current_state)
                    logging.debug("Piece to place: ",
                                  self.current_state.get_selected_piece())
                    logging.debug("Board: ")
                    logging.debug(self.current_state.state_as_array())
                    action = self.get_action(self.current_state)
                    self.current_state.select(selected_piece)
                    self.current_state.place(action[0], action[1])
                    self.current_state.set_selected_piece(action[2])
                    self.current_state.switch_player()
                    player = 1 - player
                else:
                    # Random moves here
                    action = random_player.place_piece()
                    next_piece = random_player.choose_piece()
                    while self.board.check_if_move_valid(self.board.get_selected_piece(), action[0], action[1], next_piece) is False:
                        action = random_player.place_piece()
                        next_piece = random_player.choose_piece()
                    self.current_state.select(
                        self.current_state.get_selected_piece())
                    self.current_state.place(action[0], action[1])
                    self.current_state.set_selected_piece(next_piece)
                    self.current_state.switch_player()
                    player = 1 - player

                if self.current_state.check_is_game_over():
                    if 1 - self.current_state.check_winner() == 0:
                        logging.info('QL-MCTS won')
                        reward = 1
                    else:
                        logging.info('Random won')
                        reward = -1
                    self.update_Q(self.previous_state, self.previous_action,
                                  reward, self.current_state)
                    break
                else:
                    self.update_Q(
                        self.previous_state, self.previous_action, reward, self.current_state)


if __name__ == '__main__':
    # load tree with MonteCarloSearchDecoder
    with open('progress.json', 'r') as f:
        tree = decode_tree(json.load(f))
    qplayer = QLearningPlayer(tree=tree)
    qplayer.train(100)

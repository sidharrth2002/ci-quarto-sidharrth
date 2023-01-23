from collections import defaultdict
from copy import deepcopy
import itertools
import json
import logging
import math
import os
import random
import time

from mcts import MonteCarloTreeSearch, MonteCarloTreeSearchDecoder, decode_tree
from quarto.objects import Quarto
from lib.players import RandomPlayer
from lib.isomorphic import BoardTransforms

import tqdm
logging.basicConfig(level=logging.DEBUG)


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
            # return random.uniform(1.0, 0.01)
            return None

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
                Q_val = self.get_Q(state, action)
                max_Q = max(max_Q, self.get_Q(state, action))
        return max_Q

    def check_if_winning_piece(self, state, piece):
        for i in range(self.BOARD_SIDE):
            for j in range(self.BOARD_SIDE):
                if state.check_if_move_valid(piece, i, j, piece):
                    cloned_state = deepcopy(state)
                    cloned_state.select(piece)
                    cloned_state.place(i, j)

                    if cloned_state.check_is_game_over():
                        print('WINNING PIECE FOUND')
                        return True, [i, j]
        return False, None

    def hardcoded_strategy_get_move(self, state):
        #  1. Play the piece handed over by the opponent:
        # (a) play a winning position if handed a winning piece;
        # (b) otherwise, play to build a line of like pieces if possible;
        # (c) otherwise, play randomly.
        # 2. Hand a piece to the opponent:
        # (a) avoid handing over a winning piece for your opponent to play;
        # (b) otherwise, choose randomly.

        board = state.state_as_array()
        selected_piece = state.get_selected_piece()
        # check if the selected piece is a winning piece
        winning_piece, position = self.check_if_winning_piece(
            state, selected_piece)
        if winning_piece:
            return selected_piece, position

        # check if the selected piece can be used to build a line of like pieces

        row_1 = [[0, 0], [0, 1], [0, 2], [0, 3]]
        # pieces in row 2
        row_2 = [[1, 0], [1, 1], [1, 2], [1, 3]]
        # pieces in row 3
        row_3 = [[2, 0], [2, 1], [2, 2], [2, 3]]
        # pieces in row 4
        row_4 = [[3, 0], [3, 1], [3, 2], [3, 3]]

        # pieces in column 1
        col_1 = [[0, 0], [1, 0], [2, 0], [3, 0]]
        # pieces in column 2
        col_2 = [[0, 1], [1, 1], [2, 1], [3, 1]]
        # pieces in column 3
        col_3 = [[0, 2], [1, 2], [2, 2], [3, 2]]
        # pieces in column 4
        col_4 = [[0, 3], [1, 3], [2, 3], [3, 3]]

        # pieces in diagonal 1
        diag_1 = [[0, 0], [1, 1], [2, 2], [3, 3]]
        # pieces in diagonal 2
        diag_2 = [[0, 3], [1, 2], [2, 1], [3, 0]]

        for line in [row_1, row_2, row_3, row_4, col_1, col_2, col_3, col_4, diag_1, diag_2]:
            # check if the selected piece can be used to build a line of like pieces
            characteristics = []
            empty_rows = []
            for el in line:
                x, y = el
                if board[x, y] != -1:
                    piece = board[x][y]
                    piece_char = state.get_piece_charachteristics(piece)
                    characteristics.append(
                        [piece_char.HIGH, piece_char.COLOURED, piece_char.SOLID, piece_char.SQUARE])
                else:
                    empty_rows.append(el)
                    characteristics.append([-1, -1, -1, -1])

            selected_piece_char = state.get_piece_charachteristics(
                selected_piece)
            selected_piece_char = [selected_piece_char.HIGH, selected_piece_char.COLOURED,
                                   selected_piece_char.SOLID, selected_piece_char.SQUARE]

            # check if characteristics has an empty row
            if [-1, -1, -1, -1] in characteristics:
                # insert the selected piece in the empty row
                empty_piece_index = characteristics.index(
                    [-1, -1, -1, -1])
                characteristics[empty_piece_index] = selected_piece_char

                # check if any column has the same characteristics
                col1 = [characteristics[0][0], characteristics[1][0],
                        characteristics[2][0], characteristics[3][0]]
                col2 = [characteristics[0][1], characteristics[1][1],
                        characteristics[2][1], characteristics[3][1]]
                col3 = [characteristics[0][2], characteristics[1][2],
                        characteristics[2][2], characteristics[3][2]]
                col4 = [characteristics[0][3], characteristics[1][3],
                        characteristics[2][3], characteristics[3][3]]

                col1 = [int(i) for i in col1]
                col2 = [int(i) for i in col2]
                col3 = [int(i) for i in col3]
                col4 = [int(i) for i in col4]

                if len(set(col1)) == 1 or len(set(col2)) == 1 or len(set(col3)) == 1 or len(set(col4)) == 1:
                    # this piece can be used to build a line of like pieces
                    logging.debug('playing to build a line of like pieces')
                    return True, list(reversed(empty_rows[-1]))

        # play randomly
        for i in range(self.BOARD_SIDE):
            for j in range(self.BOARD_SIDE):
                for next_piece in range(16):
                    if state.check_if_move_valid(selected_piece, i, j, next_piece):
                        return False, [i, j]

        print('returning nothing')
        print(state.state_as_array())
        print(state.get_selected_piece())

    def hardcoded_strategy_get_piece(self, state):
        possible_pieces = []
        for i in range(16):
            # check if the piece is a winning piece
            winning_piece, _ = self.check_if_winning_piece(state, i)
            if not winning_piece and i not in itertools.chain.from_iterable(state.state_as_array()):
                possible_pieces.append(i)

        return random.choice(possible_pieces)

    def get_action(self, state, mode='training'):
        '''
        If state, action pair not in Q, go to Monte Carlo Tree Search to find best action
        '''
        if mode == 'training':
            # exploration through epsilon greedy
            # look for good moves through Monte Carlo Tree Search
            if random.random() < self.epsilon:
                for i in range(30):
                    self.tree.do_rollout(state)
                best_action = self.tree.place_piece()
                return best_action
            else:
                # look in the q table for the best action
                expected_score = 0
                best_action = None
                for action in self.get_possible_actions(state):
                    if self.get_Q(state, action) is not None and expected_score < self.get_Q(state, action):
                        print('found in Q table')
                        expected_score = self.get_Q(state, action)
                        best_action = action
                # go to Monte Carlo Tree Search if no suitable action found in Q table
                if best_action is None or expected_score == 0:
                    logging.debug(
                        'No suitable action found in Q table, going to Monte Carlo Tree Search')
                    for i in range(30):
                        self.tree.do_rollout(state)
                    best_action = self.tree.place_piece()
                else:
                    print('found in Q table')

                return best_action
        else:
            # in test mode, use the Q table to find the best action
            # only go to Monte Carlo Tree Search if no suitable action found in Q table
            expected_score = 0
            best_action = None
            for action in self.get_possible_actions(state):
                if self.get_Q(state, action) is not None and expected_score < self.get_Q(state, action):
                    expected_score = self.get_Q(state, action)
                    best_action = action
            # go to Monte Carlo Tree Search if no suitable action found in Q table
            if best_action is None or expected_score == 0:
                logging.debug(
                    'No suitable action found in Q table, going to Monte Carlo Tree Search')
                for i in range(50):
                    self.tree.do_rollout(state)
                best_action = self.tree.place_piece()
            return best_action

    def update_Q(self, state, action, reward, next_state):
        Q_val = self.get_Q(state, action)
        if Q_val is None:
            Q_val = random.uniform(1.0, 0.01)
        self.set_Q(state, action, Q_val + self.alpha *
                   (reward + self.gamma * self.get_max_Q(next_state) - Q_val))

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
        wins = 0
        tries = 0
        agent_decision_times = []

        progress_bar = tqdm.tqdm(total=iterations)
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
                    # self.previous_state = deepcopy(self.current_state)
                    # logging.debug("Piece to place: ",
                    #               self.current_state.get_selected_piece())
                    # logging.debug("Board: ")
                    # logging.debug(self.current_state.state_as_array())
                    # time_start = time.time()
                    # action = self.get_action(self.current_state)
                    # time_end = time.time()
                    # agent_decision_times.append(time_end - time_start)
                    # self.current_state.select(selected_piece)
                    # self.current_state.place(action[0], action[1])
                    # self.current_state.set_selected_piece(action[2])
                    # self.current_state.switch_player()
                    # player = 1 - player

                    self.previous_state = deepcopy(self.current_state)
                    winning_piece, position = self.hardcoded_strategy_get_move(
                        self.current_state)
                    self.current_state.select(selected_piece)
                    self.current_state.place(position[0], position[1])
                    next_piece = self.hardcoded_strategy_get_piece(
                        self.current_state)
                    self.current_state.set_selected_piece(next_piece)
                    self.current_state.switch_player()

                    logging.debug("After QL-MCTS move")
                    logging.debug(self.current_state.state_as_array())

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
                    logging.debug("After random move")
                    logging.debug(self.current_state.state_as_array())

                if self.current_state.check_is_game_over():
                    if 1 - self.current_state.check_winner() == 0:
                        logging.info('QL-MCTS won')
                        reward = 1
                        wins += 1
                    else:
                        logging.info('Random won')
                        reward = -1
                    self.update_Q(self.previous_state, self.previous_action,
                                  reward, self.current_state)
                    break
                else:
                    self.update_Q(
                        self.previous_state, self.previous_action, reward, self.current_state)

            tries += 1
            if i % 50 == 0:
                logging.info(f'Iteration {i}')
                logging.info(f'Wins: {wins}')
                logging.info(f'Tries: {tries}')
                logging.info(f'Win rate: {wins/tries}')
                wins = 0
                tries = 0

            # clear the tree every time
            self.tree = MonteCarloTreeSearch(board=self.board)

            # if average agent decision time is too long, clear the MCTS tree
            # if sum(agent_decision_times) / len(agent_decision_times) > 5:
            #     self.tree = MonteCarloTreeSearch(board=self.board)
            #     agent_decision_times = []

            progress_bar.update(1)


if __name__ == '__main__':
    # load tree with MonteCarloSearchDecoder
    # with open('progress.json', 'r') as f:
    #     tree = decode_tree(json.load(f))
    qplayer = QLearningPlayer()
    qplayer.train(100)

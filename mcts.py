
from collections import defaultdict
import logging
import math
import random

import numpy as np
from dqn import RandomPlayer

from quarto.objects import Quarto


class Node:
    '''
    Node on tree
    '''
    def __init__(self, board: Quarto):
        self.board = board
        self.hashed_board = self.hash_state(board)

    def hash_state(self, board: Quarto):
        '''
        Hash the board state, current player and selected piece
        '''
        return board.board_to_string() + '||' + str(self.__current_player) + '||' + str(self.__selected_piece_index)

    def unhash_state(self, state):
        '''
        Unhash the board state, current player and selected piece
        '''
        board, current_player, selected_piece_index = state.split('||')
        board = self.string_to_board(board)
        current_player = int(current_player)
        selected_piece_index = int(selected_piece_index)

    def get_board(self):
        return self.unhash_state(self.hashed_board)

    def find_children(self):
        '''
        Find children for monte carlo tree search
        '''
        # if game is over return empty set
        if self.board.check_winner() >= 0 or self.board.check_finished():
            return set()

        return {
            self.board.make_move(piece, x, y, next_piece) for piece in range(self.MAX_PIECES) for x in range(self.BOARD_SIDE) for y in range(self.BOARD_SIDE) for next_piece in range(self.MAX_PIECES) if self.board.check_if_move_valid(piece, x, y, next_piece)
        }

    def reward(self):
        board = self.board
        if not board.check_is_game_over():
            logging.info("reward: game is not over, nonterminal board")
            return None
        if board.check_winner() == board.get_current_player():
            logging.info("reward: game is over, and you already won")
            return None
        if board.get_current_player() == (1 - board.get_current_player()):
            return 0
        if board.check_if_draw():
            logging.info('tie')
            return 0.5
        return None

    def find_random_child(self):
        board = self.board
        if board.check_winner() >= 0 or board.check_finished():
            return None
        possible_actions = [(x, y, piece, next_piece) for piece in range(self.MAX_PIECES) for x in range(self.BOARD_SIDE) for y in range(self.BOARD_SIDE) for next_piece in range(self.MAX_PIECES) if board.check_if_move_valid(piece, x, y, next_piece)]
        return board.make_move(*random.choice(possible_actions))

    def is_terminal(self):
        return self.board.check_is_game_over()

class MonteCarloTreeSearch:
    '''
    Solve using Monte Carlo Tree Search
    '''
    def __init__(self, epsilon=0.1, max_depth=1000):
        self.epsilon = epsilon
        self.max_depth = max_depth
        self.Q = defaultdict(int)
        self.N = defaultdict(int)

    def make_move(self, board: Quarto, x: int, y: int, piece: int, next_piece: int):
        '''
        Make move on board
        '''
        board.make_move(piece, x, y, next_piece)
        return Quarto(board.state_as_array())

    # def find_children(self, board: Quarto, player: int):
    #     '''
    #     Find children for monte carlo tree search
    #     '''
    #     # if game is over return empty set
    #     if board.check_winner() >= 0 or board.check_finished():
    #         return set()

    #     return {
    #         self.make_move(board, piece, x, y, next_piece) for piece in range(self.MAX_PIECES) for x in range(self.BOARD_SIDE) for y in range(self.BOARD_SIDE) for next_piece in range(self.MAX_PIECES) if board.check_if_move_valid(piece, x, y, next_piece)
    #     }

    # def find_random_child(self, board: Quarto):
    #     if board.check_winner() >= 0 or board.check_finished():
    #         return None
    #     possible_actions = [(x, y, piece, next_piece) for piece in range(self.MAX_PIECES) for x in range(self.BOARD_SIDE) for y in range(self.BOARD_SIDE) for next_piece in range(self.MAX_PIECES) if board.check_if_move_valid(piece, x, y, next_piece)]
    #     return board.make_move(*random.choice(possible_actions))

    # def reward(self, board: Quarto):
    #     if not board.check_is_game_over():
    #         logging.info("reward: game is not over, nonterminal board")
    #         return None
    #     if board.check_winner() == board.get_current_player():
    #         logging.info("reward: game is over, and you already won")
    #         return None
    #     if board.get_current_player() == (1 - board.get_current_player()):
    #         return 0
    #     if board.check_if_draw():
    #         logging.info('tie')
    #         return 0.5
    #     return None

    def choose(self, node):
        '''
        Choose best successor of node (move)
        '''
        if node.check_is_game_over():
            return None

        if node not in self.children:
            return node.find_random_child()

        def score(n):
            if self.N[n] == 0:
                return float('-inf')
            return self.Q[n] / self.N[n]

        return max(self.children[node], key=score)

    def do_rollout(self, board):
        '''
        Rollout from the node for one iteration
        '''
        node = Node(board)
        path = self.select(node)
        leaf = path[-1]
        self.expand(leaf)
        reward = self.simulate(leaf)
        self.backpropagate(path, reward)

    def select(self, node):
        '''
        Select path to leaf node
        '''
        path = []
        while True:
            path.append(node)
            if node not in self.children or not self.children[node]:
                return path
            unexplored = self.children[node] - self.children.keys()
            if unexplored:
                n = unexplored.pop()
                path.append(n)
                return path
            node = self.uct_select(node)

    def expand(self, node):
        if node in self.children:
            return
        self.children[node] = node.find_children()

    def simulate(self, node):
        '''
        Returns reward for random simulation
        '''
        invert_reward = True
        while True:
            if node.is_terminal():
                reward = node.reward()
                return 1 - reward if invert_reward else reward
            node = node.find_random_child()
            invert_reward = not invert_reward

    def backpropagate(self, path, reward):
        '''
        Backpropagate reward
        '''
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward
            reward = 1 - reward

    def uct_select(self, node):
        '''
        Select a child of node, balancing exploration & exploitation
        '''
        log_N_vertex = math.log(self.N[node])

        def uct(n):
            return self.Q[n] / self.N[n] + self.epsilon * math.sqrt(log_N_vertex / self.N[n])

        return max(self.children[node], key=uct)

    def find_random_child(self, board: Quarto):
        if board.check_winner() >= 0 or board.check_finished():
            return None
        possible_actions = [(x, y, piece, next_piece) for piece in range(self.MAX_PIECES) for x in range(self.BOARD_SIDE) for y in range(self.BOARD_SIDE) for next_piece in range(self.MAX_PIECES) if board.check_if_move_valid(piece, x, y, next_piece)]
        return board.make_move(*random.choice(possible_actions))

    def reward(self, board: Quarto):
        if not board.check_is_game_over():
            return 0

# Training while playing
tree = MonteCarloTreeSearch()
board = Quarto()
random_player = RandomPlayer()
chosen_piece = random_player.choose_piece(board)
while True:
    # random player moves
    chosen_location = random_player.place_piece(board, chosen_piece)
    board.select(chosen_piece)
    board.place(**chosen_location)
    if board.check_is_game_over():
        break
    # monte carlo tree search moves
    for _ in range(20):
        tree.do_rollout(board)
    board = tree.choose(board)
    if board.check_is_game_over():
        break

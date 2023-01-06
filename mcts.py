"""
Sidharrth Nagappan

Monte Carlo Tree Search
"""

from collections import defaultdict
import copy
import logging
import math
import pickle
import random

import numpy as np
from dqn import RandomPlayer

from quarto.objects import Quarto

logging.basicConfig(level=logging.INFO)


class Node:
    '''
    Node on tree
    '''

    def __init__(self, board: Quarto):
        self.board = copy.deepcopy(board)
        self.hashed_board = self.hash_state()
        self.MAX_PIECES = 16
        self.BOARD_SIDE = 4
        self.selected_piece = None

    def hash_state(self):
        '''
        Hash the board state, current player and selected piece
        '''
        board = self.board
        return board.board_to_string() + '||' + str(board.get_current_player()) + '||' + str(board.get_selected_piece())

    def string_to_board(self, string):
        board = np.zeros((self.BOARD_SIDE, self.BOARD_SIDE))
        i = 0
        for row in board:
            for j in range(len(row)):
                row[j] = int(string[i])
                i += 2
        return board

    def unhash_state(self, state):
        '''
        Unhash the board state, current player and selected piece
        '''
        board, current_player, selected_piece_index = state.split('||')
        board = self.string_to_board(board)
        current_player = int(current_player)
        selected_piece_index = int(selected_piece_index)
        return board, current_player, selected_piece_index

    def get_board(self):
        return self.unhash_state(self.hashed_board)

    def find_children(self):
        '''
        Find children for monte carlo tree search
        '''
        # if game is over return empty set
        if self.board.check_is_game_over():
            return set()

        new_stuff = {
            create_node(self.board.make_move(self.board.get_selected_piece(), x, y, next_piece, newboard=True)) for x in range(self.BOARD_SIDE) for y in range(self.BOARD_SIDE) for next_piece in range(self.MAX_PIECES) if self.board.check_if_move_valid(self.board.get_selected_piece(), x, y, next_piece)
        }

        return new_stuff

    def reward(self):
        board = self.board
        logging.debug("WINNER: ", board.check_winner())
        logging.debug("PLAYER: ", board.get_current_player())
        logging.debug(board)
        if not board.check_is_game_over():
            raise RuntimeError("reward called on non-terminal node")
        # logging.debug("WINNER: ", board.check_winner())
        if 1 - board.check_winner() == board.get_current_player():
            logging.debug("game is over, and you already won")
            raise RuntimeError("reward called on unreachable node")
        if board.check_winner() == board.get_current_player():
            logging.debug("other guy won")
            return 0
        if board.check_if_draw():
            logging.debug('tie')
            return 0.5
        raise RuntimeError("nothing works")
        # return 0

    def find_random_child(self):
        board = self.board
        if board.check_is_game_over():
            return None
        pa = [[board.get_selected_piece(), x, y, next_piece] for x in range(self.BOARD_SIDE) for y in range(self.BOARD_SIDE)
              for next_piece in range(self.MAX_PIECES) if board.check_if_move_valid(board.get_selected_piece(), x, y, next_piece)]
        pa = random.choice(pa)
        return create_node(board.make_move(pa[0], pa[1], pa[2], pa[3], newboard=True))

    def is_terminal(self):
        return self.board.check_is_game_over()

    def __hash__(self):
        return hash(self.hash_state())

    def __eq__(self, other):
        return self.hash_state() == other.hash_state() and self.board.get_current_player() == other.board.get_current_player() and self.board.get_selected_piece() == other.board.get_selected_piece()


def create_node(content):
    return Node(content)


class MinMaxPlayer:
    '''
    Min Max Player
    '''

    def __init__(self, depth=2):
        self.depth = depth

    def minmax(self, node, depth, maximizingPlayer, alpha, beta):
        if depth == 0 or node.is_terminal():
            return node.reward()
        if maximizingPlayer:
            v = -math.inf
            for n in node.find_children():
                v = max(v, self.minmax(n, depth - 1, False, alpha, beta))
                alpha = max(alpha, v)
                if beta <= alpha:
                    break
            return v
        else:
            v = math.inf
            for n in node.find_children():
                v = min(v, self.minmax(n, depth - 1, True, alpha, beta))
                beta = min(beta, v)
                if beta <= alpha:
                    break
            return v

    def play(self, board):
        '''
        Choose the best move
        '''
        possible_moves = []
        for x in range(4):
            for y in range(4):
                for next_piece in range(16):
                    if board.check_if_move_valid(board.get_selected_piece(), x, y, next_piece):
                        possible_moves.append([board.get_selected_piece(), x, y, next_piece, self.minmax(create_node(board.make_move(
                            board.get_selected_piece(), x, y, next_piece, newboard=True)), self.depth, False, -math.inf, math.inf)])
        best_move = max(possible_moves, key=lambda x: x[4])[0:4]
        return board.make_move(best_move[0], best_move[1], best_move[2], best_move[3], newboard=True)


class MonteCarloTreeSearch:
    '''
    Solve using Monte Carlo Tree Search
    '''

    def __init__(self, epsilon=0.1, max_depth=1000):
        self.epsilon = epsilon
        self.max_depth = max_depth
        self.Q = defaultdict(int)
        self.N = defaultdict(int)
        self.children = dict()
        self.MAX_PIECES = 16
        self.BOARD_SIDE = 4

    def choose(self, node):
        '''
        Choose best successor of node (move)
        '''
        node = Node(node)
        if node.is_terminal():
            logging.debug(node.state_as_array())
            raise RuntimeError("choose called on terminal node")

        if node not in self.children:
            return node.find_random_child()

        def score(n):
            logging.debug(f"Before reading in choose {n}")
            if self.N[n] == 0:
                return float('-inf')
            return self.Q[n] / self.N[n]

        return max(self.children[node], key=score).board

    def do_rollout(self, board):
        '''
        Rollout from the node for one iteration
        '''
        logging.debug("Rollout")
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
        # print('Expanding')
        if node in self.children:
            return
        self.children[node] = node.find_children()
        # print('Children: ', self.children[node])

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
        logging.debug('Backpropagating')
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward
            reward = 1 - reward

    def uct_select(self, node):
        '''
        Select a child of node, balancing exploration & exploitation
        '''
        assert all(n in self.children for n in self.children[node])

        log_N_vertex = math.log(self.N[node])

        def uct(n):
            return self.Q[n] / self.N[n] + self.epsilon * math.sqrt(log_N_vertex / self.N[n])

        return max(self.children[node], key=uct)

    def save_progress(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def load_progress(self, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)


logging.info("Training")
# Training while playing
tree = MonteCarloTreeSearch()
for i in range(100):
    board = Quarto()
    random_player = RandomPlayer(board)
    board.set_selected_piece(random_player.choose_piece(board))
    logging.info(f"Iteration: {i}")
    while True:
        # random player moves
        chosen_location = random_player.place_piece(
            board, board.get_selected_piece())
        chosen_piece = random_player.choose_piece(board)
        while not board.check_if_move_valid(board.get_selected_piece(), chosen_location[0], chosen_location[1], chosen_piece):
            chosen_location = random_player.place_piece(
                board, board.get_selected_piece())
            chosen_piece = random_player.choose_piece(board)
        board.select(board.get_selected_piece())
        board.place(chosen_location[0], chosen_location[1])
        board.set_selected_piece(chosen_piece)
        board.switch_player()
        if board.check_is_game_over():
            if 1 - board.check_winner() == 0:
                logging.info("Random player won")
            else:
                logging.info("Draw")
            break
        # monte carlo tree search moves
        for _ in range(50):
            tree.do_rollout(board)
        board = tree.choose(board)
        if board.check_is_game_over():
            # TODO: check if it's a draw
            if 1 - board.check_winner() == 1:
                logging.info("Monte Carlo Tree Search won")
            else:
                logging.info("Draw")
            break
        # don't need to switch player because it's done in choose
        # random_player needs to do it because it is not done automatically

        # save progress every 20 iterations
        if i % 20 == 0:
            logging.debug("Saving progress")
            tree.save_progress("progress.pkl")

logging.info("Testing")
for i in range(100):
    board = Quarto()
    while True:
        # random player moves
        chosen_location = random_player.place_piece(
            board, board.get_selected_piece())
        chosen_piece = random_player.choose_piece(board)
        while not board.check_if_move_valid(board.get_selected_piece(), chosen_location[0], chosen_location[1], chosen_piece):
            chosen_location = random_player.place_piece(
                board, board.get_selected_piece())
            chosen_piece = random_player.choose_piece(board)
        board.select(board.get_selected_piece())
        board.place(chosen_location[0], chosen_location[1])
        board.set_selected_piece(chosen_piece)
        if board.check_is_game_over():
            logging.info("Random player won")
            break
        logging.debug("After random player move: ")
        logging.debug(board.state_as_array())
        board.switch_player()
        # monte carlo tree search moves
        board = tree.choose(board)
        logging.debug("After monte carlo tree search move: ")
        logging.debug(board.state_as_array())
        if board.check_is_game_over():
            logging.info("Monte Carlo Tree Search won")
            break

# print(tree.Q.values())
# winner = board.get_current_player()

# print("-----------------")
# print("PARENT")
# parent = list(tree.children.keys())[-1]
# print(parent.board.state_as_array())
# for child in tree.children[list(tree.children.keys())[-1]]:
#     print("CHILD")
#     print(child.board.state_as_array())

# print("Testing")

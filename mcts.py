"""
Sidharrth Nagappan

Monte Carlo Tree Search
"""

from collections import defaultdict
import copy
import json
import logging
import math
import pickle
import random
from threading import Thread

import numpy as np
from lib.isomorphic import BoardTransforms
from lib.players import Player, RandomPlayer
from lib.utilities import Node, NodeDecoder, NodeEncoder

from quarto.objects import Quarto

logging.basicConfig(level=logging.INFO)


class MonteCarloTreeSearchEncoder(json.JSONEncoder):
    def default(self, obj):
        l = {
            'Q': {k.hash_state(): v for k, v in obj.Q.items()},
            'N': {k.hash_state(): v for k, v in obj.N.items()},

            # children is a dictionary of nodes
            'children': {k.hash_state(): [NodeEncoder().default(i) for i in v] for k, v in obj.children.items()},

            # 'children': [NodeEncoder().default(child) for child in obj.children],
            'epsilon': obj.epsilon,
        }
        return l

    def encode(self, obj):
        return super().encode(obj)

    def load_json(self, filename):
        with open(filename, 'r') as f:
            return json.load(f, cls=MonteCarloTreeSearchDecoder)


class MonteCarloTreeSearchDecoder(json.JSONDecoder):
    '''
    Recreate MonteCarloTreeSearch object from JSON
    '''

    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(
            self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        children = {}

        for k, v in obj['children'].items():
            children[Node(hashed_state=k)] = [
                NodeDecoder().object_hook(node) for node in v]

        if 'Q' in obj:
            return MonteCarloTreeSearch(
                Q={Node(hashed_state=k): v for k, v in obj['Q'].items()},
                N={Node(hashed_state=k): v for k, v in obj['N'].items()},
                children=children,
                epsilon=obj['epsilon'],
            )
        return obj


def decode_tree(tree):
    return MonteCarloTreeSearchDecoder().object_hook(tree)


class MonteCarloTreeSearch(Player):
    '''
    Solve using Monte Carlo Tree Search
    '''

    def __init__(self, board=Quarto(), epsilon=0.1, max_depth=1000, Q=None, N=None, children=None):
        self.epsilon = epsilon
        self.max_depth = max_depth
        if Q is None:
            self.Q = defaultdict(int)
        else:
            self.Q = Q
        if N is None:
            self.N = defaultdict(int)
        else:
            self.N = N
        if children is None:
            self.children = dict()
        else:
            self.children = children
        self.MAX_PIECES = 16
        self.BOARD_SIDE = 4
        self.board = board
        self.random_factor = 0
        self.decisions = 0
        super().__init__(board)

    def set_board(self, board):
        self.board = board

    def choose(self, node):
        '''
        Choose best successor of node (move)
        Returns the board itself
        '''
        def score(n):
            logging.debug(f"Before reading in choose {n}")
            if self.N[n] == 0:
                return float('-inf')
            return self.Q[n] / self.N[n]

        node = Node(node)
        if node.is_terminal():
            logging.debug(node.board.state_as_array())
            raise RuntimeError("choose called on terminal node")

        # number of moves made in game
        self.decisions += 1

        for key in self.children:
            if key == node:
                return max(self.children[key], key=score).board

        self.random_factor += 1
        if node not in self.children:
            for key, value in self.children.items():
                if BoardTransforms().compare_boards(node.board.state_as_array(), key.board.state_as_array()):
                    if key in self.children:
                        print("found in symmetry")
                        return max(self.children[key], key=score).board

            # number of times have to resort to random
            rand_child = node.find_random_child()
            # add to children
            self.children[node] = [rand_child]
            return rand_child.board

        print("found in board")
        return max(self.children[node], key=score).board

    def choose_piece(self):
        '''
        Choose a piece to make the opponent place
        '''
        node = Node(board=self.board,
                    selected_piece_index=self.board.get_selected_piece())

        if node.is_terminal():
            logging.debug(node.board.state_as_array())
            raise RuntimeError("choose called on terminal node")

        if node not in self.children:
            # index -1 of tuple is next piece from a board
            return node.find_random_child()[-1]

        def score(n):
            logging.debug(f"Before reading in choose {n}")
            if self.N[n] == 0:
                return float('-inf')
            return self.Q[n] / self.N[n]

        return max(self.children[node], key=score)[-1]

    def place_piece(self):
        '''
        Return position to place piece on board
        '''
        node = Node(board=self.board,
                    selected_piece_index=self.board.get_selected_piece())

        if node.is_terminal():
            logging.debug(node.board.state_as_array())
            raise RuntimeError("choose called on terminal node")

        if node not in self.children:
            piece, x, y, next_piece = node.find_random_child().move
            return x, y, next_piece

        def score(n):
            logging.debug(f"Before reading in choose {n}")
            if self.N[n] == 0:
                return float('-inf')
            return self.Q[n] / self.N[n]

        print("In place piece")
        print(max(self.children[node], key=score).move)
        return max(self.children[node], key=score).move[1:]

    def do_rollout(self, board):
        '''
        Rollout from the node for one iteration
        '''
        logging.debug("Rollout")
        # if root node, there is no move
        node = Node(board, move=())
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
                print(path)
                return path
            unexplored = self.children[node] - self.children.keys()
            if unexplored:
                n = unexplored.pop()
                path.append(n)
                return path
            node = self.uct_select(node)

    def expand(self, node):
        # logging.debug('Expanding')
        if node in self.children:
            return
        self.children[node] = node.find_children()
        # logging.debug('Children: ', self.children[node])

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

    def train_engine(self, board, num_sims=200, save_format='json'):
        '''
        Train the model
        '''
        for i in range(num_sims):
            board = Quarto()
            random_player = RandomPlayer(board)
            board.set_selected_piece(random_player.choose_piece(board))
            logging.info(f"Iteration: {i} with tree size {len(self.children)}")
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
                # setting the piece for the next player
                board.set_selected_piece(chosen_piece)
                board.switch_player()

                if board.check_is_game_over():
                    if 1 - board.check_winner() == 0:
                        logging.info("Random player won")
                    else:
                        logging.info("Draw")
                    break
                # monte carlo tree search moves

                # make move with monte carlo tree search
                for _ in range(50):
                    self.do_rollout(board)
                board = self.choose(board)

                if board.check_is_game_over():
                    # TODO: check if it's a draw
                    if 1 - board.check_winner() == 1:
                        logging.info("Agent won")
                    else:
                        logging.info("Draw")
                    break
                # don't need to switch player because it's done in choose
                # random_player needs to do it because it is not done automatically

            # print(f"Random factor ", self.random_factor / self.decisions)

            # save progress every 10 iterations
            if i % 10 == 0:
                logging.debug("Saving progress")
                if save_format == 'json':
                    self.save_progress_json('/Volumes/USB/progress.json')
                else:
                    self.save_progress_pickle('progress.pkl')

    def train(self):
        '''
        Train without multithreading
        '''
        self.train_engine(Quarto(), 100, 'json')

    def threaded_training(self, num_threads=1, save_format='json'):
        '''
        Train the model
        '''
        thread_pool = []

        for i in range(num_threads):
            t = Thread(target=self.train_engine, args=(Quarto(), 100, 'json'))
            t.start()
            thread_pool.append(t)

        for t in thread_pool:
            t.join()

        # final save after training
        if save_format == 'json':
            self.save_progress_json('progress.json')
        else:
            self.save_progress_pickle('progress.pkl')

    def generate_future_probabilities(self, root: Node, node: Node):
        '''
        play an action a from the root state st with probability proportional to the number of times that action was chosen during Phase One. To do this, AlphaGo Zero creates a probability distribution πt over the actions from the state st such that πt(a) ∝ N(st,a)^-1/τ for some hyperparameter τ; when τ = 1 the distribution exactly matches the ratios of the visit counts, while when τ → 0 the probability mass focuses on the action that was chosen most often. Using this distribution to selects actions improves the performance of AlphaGo Zero because πt is a refinement of the prediction pt for the start state st; as MCTS is allowed to run, it starts selecting actions with high value estimates more frequently rather than relying on the prior probability bonus exploration term.
        '''
        # 1 is the default value, but it can be changed to 0.5 or 0.1

        self.tau = 0.5
        if node not in self.children:
            self.do_rollout(root.board)

        probs = [self.N[child] / self.N[root]
                 for child in self.children[node]]

        probs = [p ** (1 / self.tau) for p in probs]

        probs = [p / sum(probs) for p in probs]

        return probs

    def save_progress_pickle(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def save_progress_json(self, filename):
        with open(filename, 'w') as f:
            json.dump(self, f, cls=MonteCarloTreeSearchEncoder)

    def load_progress_json(self, filename):
        with open(filename, 'r') as f:
            return json.load(f, cls=MonteCarloTreeSearchDecoder)

    def load_progress(self, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)


if __name__ == "__main__":
    mcts = MonteCarloTreeSearch()
    # with open('/Volumes/USB/progress.json', 'r') as f:
    #     mcts = decode_tree(json.load(f))
    #     logging.info("Loaded progress")
    logging.info("Starting training")
    mcts.train()

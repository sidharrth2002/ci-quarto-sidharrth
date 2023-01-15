import copy
import json
import logging
import random
from threading import Lock

import numpy as np

from quarto.objects import Quarto


class Node:
    '''
    Node on tree
    '''

    def __init__(self, board: Quarto, move=None):
        self.board = copy.deepcopy(board)
        self.hashed_board = self.hash_state()
        self.MAX_PIECES = 16
        self.BOARD_SIDE = 4
        self.selected_piece = None
        # move taken to get to this node
        self.move = move
        self.lock = Lock()

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
            create_node(self.board.make_move(self.board.get_selected_piece(), x, y, next_piece, newboard=True, return_move=True)) for x in range(self.BOARD_SIDE) for y in range(self.BOARD_SIDE) for next_piece in range(self.MAX_PIECES) if self.board.check_if_move_valid(self.board.get_selected_piece(), x, y, next_piece)
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
        return create_node(board.make_move(pa[0], pa[1], pa[2], pa[3], newboard=True, return_move=True))

    def is_terminal(self):
        return self.board.check_is_game_over()

    def check_if_board_is_symmetric(self, board):
        '''
        Check if board is symmetric
        '''
        for i in range(4):
            for j in range(4):
                if board[i][j] != board[3 - i][3 - j]:
                    return False
        return True

    def get_canonical_representation(self, board):
        '''
        Get canonical representation of the board
        '''
        raise NotImplementedError("Not implemented")

    def normal_form(self, board):
        '''
        Return normal form of board
        Two boards are equivalent if they reduce to the same normal form.
        The normal form is defined by the sequence of steps that generate it, and the sequence
        is chosen to ensure that all equivalent instances are reduced to the same normal form.
        '''
        normal_form = []

        # all of the positional symmetries are applied to the original instance, generating 32 equivalent, possibly distinct instances
        # each of these instances have and associated tag
        # this tag, or positional bit-mask, is a 17-bit string in which the ith bit is set if the ith square of the board is occuped

        # those instances that share the largest bit-mask are candidates for normal form

        # each candidate instance is mapped with and XOR piece transformation.
        # the constant used is the value of the piece on the first occupied square

        # all 24 bitwise-permutaton piece transformations are applied to each of the candidate instances
        # the resulting instances are compared to each other with a string comparison
        # instance that results in the lexicographically least string is selected as the normal form
        raise NotImplementedError("Not implemented")

    def find_equivalent_boards(self, board):
        '''
        Find isomorphic boards
        '''
        raise NotImplementedError("Not implemented")

    def __hash__(self):
        return hash(self.hash_state())

    def __eq__(self, other):
        return self.hash_state() == other.hash_state() and self.board.get_current_player() == other.board.get_current_player() and self.board.get_selected_piece() == other.board.get_selected_piece()


def create_node(content):
    # board and move taken to reach this node
    return Node(content[0], content[1])


class NodeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Node):
            return {
                'board': obj.board,
                'move': obj.move,
                'selected_piece': obj.selected_piece,
            }
        return json.JSONEncoder.default(self, obj)


class NodeDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(
            self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        if 'board' in obj and 'move' in obj:
            return Node(obj['board'], obj['move'])
        return obj


class MonteCarloTreeSearchEncoder(json.JSONEncoder):
    def default(self, obj):
        return {
            'Q': obj.Q,
            'N': obj.N,
            'children': [NodeEncoder().default(child) for child in obj.children],
            'epsilon': obj.epsilon,
        }

    def load_json(self, filename):
        with open(filename, 'r') as f:
            return json.load(f, cls=MonteCarloTreeSearchDecoder)


class MonteCarloTreeSearchDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(
            self, object_hook=self.object_hook, *args, **kwargs)

    # def object_hook(self, obj):
    #     if 'Q' in obj:
    #         return MonteCarloTreeSearch(obj['Q'], obj['N'], [NodeDecoder().decode(child) for child in obj['children']], obj['epsilon'], obj['MAX_PIECES'], obj['BOARD_SIDE'])
    #     return obj

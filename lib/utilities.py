import copy
import json
import logging
import os
import random
from threading import Lock

import numpy as np

from quarto.objects import Quarto

import hashlib


class Node:
    '''
    Node on tree
    '''

    def __init__(self, board=None, move=None, current_player=None, selected_piece_index=None, hashed_state=None):
        # recreating node from a hashed state
        self.MAX_PIECES = 16
        self.BOARD_SIDE = 4

        if hashed_state is not None:
            # selected piece is the piece that is selected for the next player after this move
            board, selected_piece_index, move = self.unhash_state(
                hashed_state)
            self.board = Quarto()
            self.board.set_board(board)
            self.selected_piece = selected_piece_index
            self.board.set_selected_piece(selected_piece_index)
            # move taken to get to this node
            # format: (piece, x, y, next_piece)
            self.move = move
        else:
            self.board = copy.deepcopy(board)
            self.selected_piece = None
            # move taken to get to this node
            self.move = move

            # set current player on the board
            if current_player is not None:
                self.board.set_current_player(current_player)

            # this is the piece that is selected for the next player
            # children will place this piece in different positions
            if selected_piece_index is not None:
                self.board.set_selected_piece(selected_piece_index)
                self.selected_piece = selected_piece_index

            # also check if move contains next piece as this should be added as selected piece
            if move is not None and len(move) != 0:
                # last element is the next piece
                self.board.set_selected_piece(move[-1])
                self.selected_piece = move[-1]

            # at the end of each turn the selected piece on the board is the piece the next player will place
            # so this should already be in the board
            self.selected_piece = self.board.get_selected_piece()

    def hash_state(self, with_move=True):
        '''
        Hash the board state, current player and selected piece
        '''
        if with_move:
            board = self.board
            return board.board_to_string() + '|| ' + str(self.selected_piece) + '|| ' + str(self.move)
        else:
            board = self.board
            val = board.board_to_string() + '|| ' + str(self.selected_piece) + '|| ' + str(())
            return val

    def string_to_board(self, string):
        board_elements = string.strip().split(' ')
        board = np.zeros((self.BOARD_SIDE, self.BOARD_SIDE))
        for i in range(len(board_elements)):
            board[i // self.BOARD_SIDE][i %
                                        self.BOARD_SIDE] = int(float(board_elements[i]))
        return board

    def unhash_state(self, state):
        '''
        Unhash the board state, current player and selected piece
        '''
        board, selected_piece_index, move = state.split('||')
        board = self.string_to_board(board)
        selected_piece_index = int(selected_piece_index)
        try:
            # string to tuple
            move = eval(move)
        except:
            move = ()
        return board, selected_piece_index, move

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

        # only take max children
        # new_stuff = list(new_stuff)
        # new_stuff = new_stuff[:min(len(new_stuff), 90)]
        # new_stuff = set(new_stuff)

        return new_stuff

    def reward(self):
        board = self.board
        logging.debug("WINNER: ", board.check_winner())
        logging.debug("PLAYER: ", board.get_current_player())
        logging.debug(board)
        if not board.check_is_game_over():
            raise RuntimeError("reward called on non-terminal node")
        # logging.debug("WINNER: ", board.check_winner())

        player_who_last_moved = 1 - board.get_current_player()

        # 1 if plays second, 0 if plays first
        agent_position = 1

        if player_who_last_moved == agent_position and 1 - board.check_winner() == agent_position:
            # agent won
            return 1
        elif player_who_last_moved == 1 - agent_position and 1 - board.check_winner() == 1 - agent_position:
            # agent lost
            return 0
        elif board.check_if_draw():
            return 0.5

        # if 1 - board.check_winner() == board.get_current_player():
        #     logging.debug("game is over, and you already won")
        #     raise RuntimeError("reward called on unreachable node")
        # if 1 - board.get_current_player() == 0 and 1 - board.check_winner() == 0:
        #     logging.debug("other guy won")
        #     return 0
        # if board.check_if_draw():
        #     logging.debug('tie')
        #     return 0.5
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
        # hashlib hash to int (make sure it is deterministic)
        # val = int(hashlib.sha1(
        #     self.hash_state().encode('utf-8')).hexdigest(), 16)
        t = str(self.board.state_as_array()) + \
            str(self.board.get_selected_piece())
        val = int(hashlib.sha1(t.encode('utf-8')).hexdigest(), 16)
        return val
        # return hash(self.hash_state())

    def __eq__(self, other):
        # return self.hash_state() == other.hash_state() and self.board.get_current_player() == other.board.get_current_player() and self.board.get_selected_piece() == other.board.get_selected_piece()
        # return self.hash_state(with_move=False) == other.hash_state(with_move=False) and self.board.get_selected_piece() == other.board.get_selected_piece()
        # print(self)
        # print(other)
        # print(np.array_equal(self.board.state_as_array(),
        #       other.board.state_as_array()))
        # print(self.board.get_selected_piece() ==
        #       other.board.get_selected_piece())
        return np.array_equal(self.board.state_as_array(), other.board.state_as_array()) and self.board.get_selected_piece() == other.board.get_selected_piece()
        # return self.hash_state(with_move=False) == other.hash_state(with_move=False)


def create_node(content):
    # board and move taken to reach this node
    piece, x, y, next_piece = content[1]
    return Node(board=content[0], move=(piece, x, y, next_piece))


class NodeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Node):
            # l = {
            #     'state': obj.hash_state(),
            # }
            # print(l)
            return obj.hash_state()
        return json.JSONEncoder.default(self, obj)


class NodeDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(
            self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        # if 'state' in obj:
        #     return Node(hashed_state=obj['state'])
        # return obj
        return Node(hashed_state=obj)

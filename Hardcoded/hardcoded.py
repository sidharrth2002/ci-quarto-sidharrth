'''
Hardcoded player for Quarto
Follows risky strategy from paper:

"Developing Strategic and Mathematical Thinking via Game Play:
Programming to Investigate a Risky Strategy for Quarto"
by Peter Rowlett
'''
from copy import deepcopy
import itertools
import logging
import random

from lib.players import Player
from quarto.objects import Quarto

import sys
sys.path.insert(0, '..')

class HardcodedPlayer(Player):
    def __init__(self, quarto: Quarto = None):
        if quarto is None:
            quarto = Quarto()
        super().__init__(quarto)
        self.BOARD_SIDE = 4

    def check_if_winning_piece(self, state, piece):
        '''
        Simulate placing the piece on the board and check if the game is over
        '''

        for i in range(self.BOARD_SIDE):
            for j in range(self.BOARD_SIDE):
                if state.check_if_move_valid(piece, i, j, -100):
                    cloned_state = deepcopy(state)
                    cloned_state.select(piece)
                    cloned_state.place(i, j)

                    if cloned_state.check_is_game_over():
                        return True, [i, j]
        return False, None

    def hardcoded_strategy_get_piece(self, state):
        '''
        Returns a piece to be placed on the board
        '''
        possible_pieces = []
        for i in range(16):
            # check if the piece is a winning piece
            winning_piece, _ = self.check_if_winning_piece(state, i)
            if (not winning_piece) and (i not in list(itertools.chain.from_iterable(state.state_as_array()))) and (i != state.get_selected_piece()):
                possible_pieces.append(i)

        # if no pieces can be placed on board anymore (board full/game over), return -1
        if len(possible_pieces) == 0:
            # check if number of non-empty cells is 16
            if len([i for i in list(itertools.chain.from_iterable(state.state_as_array())) if i != -1]) == 16:
                return -1
            else:
                # there are possible pieces to be placed, but they are winning pieces/already in board
                on_board = list(itertools.chain.from_iterable(
                    state.state_as_array()))
                not_on_board = list(set(range(16)) - set(on_board))
                return random.choice(not_on_board)
        else:
            return random.choice(possible_pieces)

    def choose_piece(self):
        '''
        Returns a piece to be placed on the board
        '''
        return self.hardcoded_strategy_get_piece()

    def hardcoded_strategy_get_move(self, return_winning_piece_boolean=True):
        #  1. Play the piece handed over by the opponent:
        # (a) play a winning position if handed a winning piece;
        # (b) otherwise, play to build a line of like pieces if possible;
        # (c) otherwise, play randomly.
        # 2. Hand a piece to the opponent:
        # (a) avoid handing over a winning piece for your opponent to play;
        # (b) otherwise, choose randomly.

        state = self.get_game()

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
                    if return_winning_piece_boolean:
                        return True, list(reversed(empty_rows[-1]))
                    else:
                        move = list(reversed(empty_rows[-1]))
                        return move[0], move[1]

        # play randomly
        for i in range(self.BOARD_SIDE):
            for j in range(self.BOARD_SIDE):
                for next_piece in range(16):
                    if state.check_if_move_valid(selected_piece, i, j, next_piece):
                        if return_winning_piece_boolean:
                            return False, [i, j]
                        else:
                            return i, j

        logging.debug(f"Selected piece: {selected_piece}")
        logging.debug(f"Board: {board}")
        logging.debug('no move found')

    def place_piece(self):
        '''
        Above function sometimes necessary to return additional information
        In game, first return value is not necessary
        '''
        return self.hardcoded_strategy_get_move(return_winning_piece_boolean=False)


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

    def hardcoded_strategy_get_piece(self):
        '''
        Returns a piece to be placed on the board
        '''
        state = self.get_game()

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

    def hardcoded_strategy_get_move(self, return_winning_piece_boolean=True, return_as_tuple=False):
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
            if return_as_tuple:
                return position[0], position[1]
            else:
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
                # count how many [-1, -1, -1, -1] are in characteristics
                empty_indexes = [i for i, x in enumerate(
                    characteristics) if x == [-1, -1, -1, -1]]

                empty_rows_count = characteristics.count([-1, -1, -1, -1])
                characteristics_copy = characteristics.copy()

                # proceeding to check couplets and see if they can build triplets
                # since 2 empty rows may be present and either could create a triplet, have to choose randomly later
                potential_moves = []

                for i, index in enumerate(empty_indexes):
                    position = empty_rows[i]
                    # insert the selected piece in the empty row
                    # empty_piece_index = characteristics.index(
                    #     [-1, -1, -1, -1])
                    characteristics = characteristics_copy.copy()
                    characteristics[index] = selected_piece_char

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

                    # print(col1, col2, col3, col4)
                    def check_if_form_triplet(line):
                        # earlier we checked if we can complete a line
                        # here we check if we can form a triplet (one step away from completing a line)
                        return line.count(1) == 3 or line.count(0) == 3

                    # if len(set(col1)) == 1 or len(set(col2)) == 1 or len(set(col3)) == 1 or len(set(col4)) == 1:
                    if check_if_form_triplet(col1) or check_if_form_triplet(col2) or check_if_form_triplet(col3) or check_if_form_triplet(col4):
                        # this piece can be used to build a line of like pieces
                        logging.debug('playing to build a line of like pieces')
                        potential_moves.append(list(reversed(position)))

                    if len(potential_moves) >= 1:
                        if return_winning_piece_boolean:
                            # return True, list(reversed(empty_rows[-1]))
                            return True, random.choice(potential_moves)
                        else:
                            # move = list(reversed(empty_rows[-1]))
                            # move = list(reversed(position))
                            move = random.choice(potential_moves)
                            return move[0], move[1]

        # play randomly
        possible_moves = []
        for i in range(self.BOARD_SIDE):
            for j in range(self.BOARD_SIDE):
                for next_piece in range(16):
                    if state.check_if_move_valid(selected_piece, i, j, next_piece):
                        if return_winning_piece_boolean:
                            possible_moves.append([False, [i, j]])
                        else:
                            possible_moves.append([i, j])

        random_move = random.choice(possible_moves)
        return random_move[0], random_move[1]

        logging.debug(f"Selected piece: {selected_piece}")
        logging.debug(f"Board: {board}")
        logging.debug('no move found')

    def place_piece(self):
        '''
        Above function sometimes necessary to return additional information
        In game, first return value is not necessary
        '''
        print('playing to build a line of like pieces')
        return self.hardcoded_strategy_get_move(return_winning_piece_boolean=False, return_as_tuple=True)


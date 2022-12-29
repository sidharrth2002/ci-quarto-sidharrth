# Free for personal or classroom use; see 'LICENSE.md' for details.
# https://github.com/squillero/computational-intelligence

import itertools
import logging
import random
import numpy as np
from abc import abstractmethod
import copy
from gym import spaces

from quarto.objects import *

class Player(object):

    def __init__(self, quarto) -> None:
        self.__quarto = quarto

    @abstractmethod
    def choose_piece(self) -> int:
        pass

    @abstractmethod
    def place_piece(self):
        pass

    def get_game(self):
        return self.__quarto


class Piece(object):

    def __init__(self, high: bool, coloured: bool, solid: bool, square: bool):
        self.HIGH = high
        self.COLOURED = coloured
        self.SOLID = solid
        self.SQUARE = square


class Quarto(object):

    MAX_PLAYERS = 2
    BOARD_SIDE = 4
    MAX_PIECES = 16

    def __init__(self, pieces = None) -> None:
        self.__board = np.ones(
            shape=(self.BOARD_SIDE, self.BOARD_SIDE), dtype=int) * -1
        self.__pieces = []
        self.__pieces.append(Piece(False, False, False, False))  # 0
        self.__pieces.append(Piece(False, False, False, True))  # 1
        self.__pieces.append(Piece(False, False, True, False))  # 2
        self.__pieces.append(Piece(False, False, True, True))  # 3
        self.__pieces.append(Piece(False, True, False, False))  # 4
        self.__pieces.append(Piece(False, True, False, True))  # 5
        self.__pieces.append(Piece(False, True, True, False))  # 6
        self.__pieces.append(Piece(False, True, True, True))  # 7
        self.__pieces.append(Piece(True, False, False, False))  # 8
        self.__pieces.append(Piece(True, False, False, True))  # 9
        self.__pieces.append(Piece(True, False, True, False))  # 10
        self.__pieces.append(Piece(True, False, True, True))  # 11
        self.__pieces.append(Piece(True, True, False, False))  # 12
        self.__pieces.append(Piece(True, True, False, True))  # 13
        self.__pieces.append(Piece(True, True, True, False))  # 14
        self.__pieces.append(Piece(True, True, True, True))  # 15
        self.__current_player = 0
        self.__players = ()
        self.__selected_piece_index = -1
        if pieces is not None:
            self.__pieces = pieces

    def set_players(self, players):
        self.__players = players

    def get_players(self):
        return self.__players

    def select(self, pieceIndex: int) -> bool:
        '''
        select a piece. Returns True on success
        '''
        if pieceIndex not in self.__board:
            self.__selected_piece_index = pieceIndex
            return True
        return False

    def place(self, x: int, y: int) -> bool:
        '''
        Place piece in coordinates (x, y). Returns true on success
        '''
        if self.__placeable(x, y):
            self.__board[y, x] = self.__selected_piece_index
            return True
        logging.debug("Invalid move: ", x, y)
        return False

    def __placeable(self, x: int, y: int) -> bool:
        return not (y < 0 or x < 0 or x > 3 or y > 3 or self.__board[y, x] >= 0)

    def print(self):
        '''
        Print the board
        '''
        for row in self.__board:
            print("\n -------------------")
            print("|", end="")
            for element in row:
                print(f" {element: >2}", end=" |")
        print("\n -------------------\n")
        print(f"Selected piece: {self.__selected_piece_index}\n")

    def get_piece_charachteristics(self, index: int) -> Piece:
        '''
        Gets charachteristics of a piece (index-based)
        '''
        return copy.deepcopy(self.__pieces[index])

    def get_board_status(self) -> np.ndarray:
        '''
        Get the current board status (pieces are represented by index)
        '''
        return copy.deepcopy(self.__board)

    def get_selected_piece(self) -> int:
        '''
        Get index of selected piece
        '''
        return copy.deepcopy(self.__selected_piece_index)

    def __check_horizontal(self) -> int:
        for i in range(self.BOARD_SIDE):
            high_values = [
                elem for elem in self.__board[i] if elem >= 0 and self.__pieces[elem].HIGH
            ]
            coloured_values = [
                elem for elem in self.__board[i] if elem >= 0 and self.__pieces[elem].COLOURED
            ]
            solid_values = [
                elem for elem in self.__board[i] if elem >= 0 and self.__pieces[elem].SOLID
            ]
            square_values = [
                elem for elem in self.__board[i] if elem >= 0 and self.__pieces[elem].SQUARE
            ]
            low_values = [
                elem for elem in self.__board[i] if elem >= 0 and not self.__pieces[elem].HIGH
            ]
            noncolor_values = [
                elem for elem in self.__board[i] if elem >= 0 and not self.__pieces[elem].COLOURED
            ]
            hollow_values = [
                elem for elem in self.__board[i] if elem >= 0 and not self.__pieces[elem].SOLID
            ]
            circle_values = [
                elem for elem in self.__board[i] if elem >= 0 and not self.__pieces[elem].SQUARE
            ]
            if len(high_values) == self.BOARD_SIDE or len(
                    coloured_values
            ) == self.BOARD_SIDE or len(solid_values) == self.BOARD_SIDE or len(
                    square_values) == self.BOARD_SIDE or len(low_values) == self.BOARD_SIDE or len(
                        noncolor_values) == self.BOARD_SIDE or len(
                            hollow_values) == self.BOARD_SIDE or len(
                                circle_values) == self.BOARD_SIDE:
                return self.__current_player
        return -1

    def __check_vertical(self):
        for i in range(self.BOARD_SIDE):
            high_values = [
                elem for elem in self.__board[:, i] if elem >= 0 and self.__pieces[elem].HIGH
            ]
            coloured_values = [
                elem for elem in self.__board[:, i] if elem >= 0 and self.__pieces[elem].COLOURED
            ]
            solid_values = [
                elem for elem in self.__board[:, i] if elem >= 0 and self.__pieces[elem].SOLID
            ]
            square_values = [
                elem for elem in self.__board[:, i] if elem >= 0 and self.__pieces[elem].SQUARE
            ]
            low_values = [
                elem for elem in self.__board[:, i] if elem >= 0 and not self.__pieces[elem].HIGH
            ]
            noncolor_values = [
                elem for elem in self.__board[:, i] if elem >= 0 and not self.__pieces[elem].COLOURED
            ]
            hollow_values = [
                elem for elem in self.__board[:, i] if elem >= 0 and not self.__pieces[elem].SOLID
            ]
            circle_values = [
                elem for elem in self.__board[:, i] if elem >= 0 and not self.__pieces[elem].SQUARE
            ]
            if len(high_values) == self.BOARD_SIDE or len(
                    coloured_values
            ) == self.BOARD_SIDE or len(solid_values) == self.BOARD_SIDE or len(
                    square_values) == self.BOARD_SIDE or len(low_values) == self.BOARD_SIDE or len(
                        noncolor_values) == self.BOARD_SIDE or len(
                            hollow_values) == self.BOARD_SIDE or len(
                                circle_values) == self.BOARD_SIDE:
                return self.__current_player
        return -1

    def __check_diagonal(self):
        high_values = []
        coloured_values = []
        solid_values = []
        square_values = []
        low_values = []
        noncolor_values = []
        hollow_values = []
        circle_values = []
        for i in range(self.BOARD_SIDE):
            if self.__board[i, i] < 0:
                break
            if self.__pieces[self.__board[i, i]].HIGH:
                high_values.append(self.__board[i, i])
            else:
                low_values.append(self.__board[i, i])
            if self.__pieces[self.__board[i, i]].COLOURED:
                coloured_values.append(self.__board[i, i])
            else:
                noncolor_values.append(self.__board[i, i])
            if self.__pieces[self.__board[i, i]].SOLID:
                solid_values.append(self.__board[i, i])
            else:
                hollow_values.append(self.__board[i, i])
            if self.__pieces[self.__board[i, i]].SQUARE:
                square_values.append(self.__board[i, i])
            else:
                circle_values.append(self.__board[i, i])
        if len(high_values) == self.BOARD_SIDE or len(coloured_values) == self.BOARD_SIDE or len(
                solid_values) == self.BOARD_SIDE or len(square_values) == self.BOARD_SIDE or len(
                    low_values
        ) == self.BOARD_SIDE or len(noncolor_values) == self.BOARD_SIDE or len(
                    hollow_values) == self.BOARD_SIDE or len(circle_values) == self.BOARD_SIDE:
            return self.__current_player
        high_values = []
        coloured_values = []
        solid_values = []
        square_values = []
        low_values = []
        noncolor_values = []
        hollow_values = []
        circle_values = []
        for i in range(self.BOARD_SIDE):
            if self.__board[i, self.BOARD_SIDE - 1 - i] < 0:
                break
            if self.__pieces[self.__board[i, self.BOARD_SIDE - 1 - i]].HIGH:
                high_values.append(self.__board[i, self.BOARD_SIDE - 1 - i])
            else:
                low_values.append(self.__board[i, self.BOARD_SIDE - 1 - i])
            if self.__pieces[self.__board[i, self.BOARD_SIDE - 1 - i]].COLOURED:
                coloured_values.append(
                    self.__board[i, self.BOARD_SIDE - 1 - i])
            else:
                noncolor_values.append(
                    self.__board[i, self.BOARD_SIDE - 1 - i])
            if self.__pieces[self.__board[i, self.BOARD_SIDE - 1 - i]].SOLID:
                solid_values.append(self.__board[i, self.BOARD_SIDE - 1 - i])
            else:
                hollow_values.append(self.__board[i, self.BOARD_SIDE - 1 - i])
            if self.__pieces[self.__board[i, self.BOARD_SIDE - 1 - i]].SQUARE:
                square_values.append(self.__board[i, self.BOARD_SIDE - 1 - i])
            else:
                circle_values.append(self.__board[i, self.BOARD_SIDE - 1 - i])
        if len(high_values) == self.BOARD_SIDE or len(coloured_values) == self.BOARD_SIDE or len(
                solid_values) == self.BOARD_SIDE or len(square_values) == self.BOARD_SIDE or len(
                    low_values
        ) == self.BOARD_SIDE or len(noncolor_values) == self.BOARD_SIDE or len(
                    hollow_values) == self.BOARD_SIDE or len(circle_values) == self.BOARD_SIDE:
            return self.__current_player
        return -1

    def check_winner(self) -> int:
        '''
        Check who is the winner
        '''
        l = [self.__check_horizontal(), self.__check_vertical(),
             self.__check_diagonal()]
        print("options: ", l)
        for elem in l:
            if elem >= 0:
                return elem
        return -1

    def check_is_game_over(self) -> bool:
        '''
        Check if the game is over
        '''
        print("Board: ", self.__board)
        print("Winner: ", self.check_winner())
        print("Finished: ", self.check_finished())
        return self.check_winner() >= 0 or self.check_finished()


    def check_finished(self) -> bool:
        '''
        Check if the game is finished
        '''
        for row in self.__board:
            for elem in row:
                if elem == -1:
                    return False
        return True

    def run(self) -> int:
        '''
        Run the game (with output for every move)
        '''
        winner = -1
        while winner < 0 and not self.check_finished():
            logging.info("Player ", self.__current_player, "turn")
            # self.print()
            piece_ok = False
            while not piece_ok:
                piece_ok = self.select(self.__players[self.__current_player].choose_piece(
                    self.__board, self.__selected_piece_index))
            piece_ok = False
            self.__current_player = (
                self.__current_player + 1) % self.MAX_PLAYERS
            # self.print()
            while not piece_ok:
                x, y = self.__players[self.__current_player].place_piece(
                    self.__board, self.__selected_piece_index)
                piece_ok = self.place(x, y)
            # print(self.state())
            winner = self.check_winner()
        # self.print()
        return winner

    def state(self) -> str:
        '''
        Return the state of the game
        '''
        return str(self.__board)

    def state_as_array(self) -> np.ndarray:
        '''
        Return the state of the game as a numpy array
        '''
        return self.__board

    def get_current_player(self):
        '''
        Return the current player
        '''
        return self.__current_player

    def switch_player(self):
        '''
        Switch the current player
        '''
        self.__current_player = 1 - self.__current_player

    def check_if_draw(self):
        '''
        Check if the game is a draw
        '''
        return self.check_finished() and self.check_winner() < 0

    def check_if_move_valid(self, piece: int, x: int, y: int, next_piece: int):
        '''
        Check if a move is valid
        '''
        # piece out of range
        if piece < 0 or piece >= self.MAX_PIECES:
            logging.debug("piece out of range")
            return False
        if x < 0 or x >= self.BOARD_SIDE:
            logging.debug("x out of range")
            return False
        if y < 0 or y >= self.BOARD_SIDE:
            logging.debug("y out of range")
            return False
        # move to position already occupied
        if self.__board[y, x] > -1:
            logging.debug("move to position already occupied")
            return False
        # if the next piece chosen is empty, then the move is invalid
        # if self.__pieces[next_piece] == -1:
        #     return False
        # chosen piece already in the board
        if next_piece in list(itertools.chain(*self.__board)):
            logging.debug("chosen piece already in the board")
            return False
        return True

    def make_move(self, piece: int, x: int, y: int, next_piece: int):
        '''
        Make a move
        '''
        if self.check_if_move_valid(piece, x, y, next_piece):
            self.__board[y, x] = piece
            self.__selected_piece_index = next_piece
            self.__current_player = (self.__current_player + 1) % self.MAX_PLAYERS
            return self
        return self

    def get_pieces(self):
        '''
        Return the pieces
        '''
        return self.__pieces

    def board_to_string(self):
        string = ''
        for row in self.__board:
            for elem in row:
                string += str(elem) + ' '
        return string

    def string_to_board(self, string):
        board = np.zeros((self.BOARD_SIDE, self.BOARD_SIDE))
        i = 0
        for row in board:
            for j in range(len(row)):
                row[j] = int(string[i])
                i += 2
        return board




class RandomPlayer(Player):
    """Random player"""

    def __init__(self, quarto: Quarto):
        super().__init__(quarto)

    def choose_piece(self, state=None):
        return random.randint(0, 15)

    def place_piece(self, state=None):
        return random.randint(0, 3), random.randint(0, 3)


class QuartoScape:
    '''Custom gym environment for Quarto'''

    def __init__(self):
        self.game = Quarto()
        self.action_space = spaces.MultiDiscrete([4, 4, 16])
        self.observation_space = spaces.MultiDiscrete([17] * 17)
        self.reward_range = (-1, 1)
        self.main_player = None

    def set_main_player(self, player):
        self.main_player = player
        self.game.set_players((player, RandomPlayer(self.game)))
        return True

    def score(self, state):
        self.gb = self.change_representation(state)
        sum = 0
        for i in range(4):
            sum_plane = 0
            for j in range(4):
                sum_row = 0
                for k in range(4):
                    sum_row += self.gb[j, k, i]
                if(abs(sum_row) == 4):
                    sum_plane += 1
                elif(abs(sum_row) == 3):
                    sum_plane += 0.7
                elif(abs(sum_row) == 2):
                    sum_plane += 0.4
                elif(abs(sum_row) == 1):
                    sum_plane += 0.1
            for k in range(4):
                sum_col = 0
                for j in range(4):
                    sum_col += self.gb[j, k, i]
                if(abs(sum_col) == 4):
                    sum_plane += 1
                elif(abs(sum_col) == 3):
                    sum_plane += 0.7
                elif(abs(sum_col) == 2):
                    sum_plane += 0.4
                elif(abs(sum_col) == 1):
                    sum_plane += 0.1
            sum += sum_plane
        # now diagonals
        for i in range(4):
            sum_diaga = 0
            sum_diagb = 0
            for k in range(4):
                sum_diaga += self.gb[k, k, i]
                sum_diagb += self.gb[k, 3-k, i]
            if(abs(sum_diaga) == 4):
                sum += 1
            elif(abs(sum_diaga) == 3):
                sum += 0.7
            elif(abs(sum_diaga) == 2):
                sum += 0.4
            elif(abs(sum_diaga) == 1):
                sum += 0.1
            # second diagonal
            if(abs(sum_diagb) == 4):
                sum += 1
            elif(abs(sum_diagb) == 3):
                sum += 0.7
            elif(abs(sum_diagb) == 2):
                sum += 0.4
            elif(abs(sum_diagb) == 1):
                sum += 0.1
        return sum

    def reward(self, state, piece, action):
        sum = 0
        score_before_action = self.score(state)
        cloned_quarto = Quarto(pieces=state)
        cloned_quarto.select(piece)
        cloned_quarto.place(action[0], action[1])
        score_after_action = self.score(cloned_quarto.state_as_array())
        return score_after_action - score_before_action

    def change_representation(self, state):
        '''
        Each piece has 4 dimensions (high, coloured, solid, square)
        '''
        new_rep = np.zeros((4, 4, 4))
        for i in range(4):
            for j in range(4):
                piece = state[i, j]
                if piece == -1:
                    continue
                piece = self.game.get_pieces()[piece]
                high = piece.HIGH
                coloured = piece.COLOURED
                solid = piece.SOLID
                square = piece.SQUARE
                new_rep[i, j, 0] = high
                new_rep[i, j, 1] = coloured
                new_rep[i, j, 2] = solid
                new_rep[i, j, 3] = square
        return new_rep

    # def score(self):
    #     sum = 0
    #     board = self.game.state_as_array()
    #     sum_plane = 0
    #     for i in range(4):
    #         sum_row = 0
    #         for j in range(4):
    #             sum_row += board[i, j]
    #         sum_plane += sum_row

    #     for j in range(4):
    #         sum_col = 0
    #         for i in range(4):
    #             sum_col += board[i, j]
    #         sum_plane += sum_col

    #     sum += sum_plane

    #     for i in range(4):
    #         sum_diaga = 0
    #         sum_diagb = 0
    #         for

    #     return sum_plane

    def step(self, action, chosen_piece):
        # position is the position the previous piece should be moved to
        # chosen next piece is the piece the agent chooses for the next player to move
        x, y, chosen_next_piece = action
        self.next_piece = chosen_next_piece
        reward = 0
        if self.game.check_if_move_valid(chosen_piece, x, y, chosen_next_piece):
            logging.info(f"Valid move, piece {chosen_piece} placed at {x}, {y}")
            self.game.select(chosen_piece)
            self.game.place(x, y)
            # self.game.print()
            if self.game.check_winner() != -1:
                # this move resulted in a win
                # reward = self.reward(self.game.state_as_array(), chosen_piece, action)
                # bonus
                reward = 1
                return self.game.state_as_array(), reward, self.game.check_is_game_over(), {}
            # elif self.game.check_if_draw():
            #     # this move resulted in a draw
            #     reward = 0.5
            #     return self.game.state_as_array(), self.game.check_winner(), self.game.check_finished(), {}
            else:
                # this move did not result in a win or a draw
                # print('Nothing happened, reward is 0')
                reward = self.reward(self.game.state_as_array(), chosen_piece, action)
                return self.game.state_as_array(), reward, self.game.check_is_game_over(), {}
        # else:
        #     print("Invalid move, fuck off")
        #     reward = -1

        return self.game.state_as_array(), reward, self.game.check_finished(), {}

    def reset(self):
        self.game = Quarto()
        self.game.set_players((self.main_player, RandomPlayer(self.game)))
        # print(self.game.state_as_array())
        return self.game.state_as_array()

    def close(self):
        print("Closing environment")

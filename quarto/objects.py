from datetime import datetime
import logging
import numpy as np
from abc import abstractmethod
import copy

from quarto.objects import *

# uncomment to see debug messages (e.g. why a move is invalid)
# logging.basicConfig(level=logging.DEBUG)

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

    def __init__(self, high: bool, coloured: bool, solid: bool, square: bool) -> None:
        self.HIGH = high
        self.COLOURED = coloured
        self.SOLID = solid
        self.SQUARE = square
        self.binary = [int(high), int(coloured), int(solid), int(square)]

class QuartoParent(object):
    '''
    Calabrese Quarto class
    '''

    MAX_PLAYERS = 2
    BOARD_SIDE = 4
    MAX_PIECES = 16

    def __init__(self, pieces=None) -> None:
        self.__players = ()
        self.reset()

    def reset(self):
        self._board = np.ones(
            shape=(self.BOARD_SIDE, self.BOARD_SIDE), dtype=int) * -1
        self._binary_board = np.full(
            shape=(self.BOARD_SIDE, self.BOARD_SIDE, 4), fill_value=np.nan)
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
        self._current_player = 0
        self.__selected_piece_index = -1

    def set_players(self, players):
        self.__players = players

    def get_current_player(self) -> int:
        '''
        Gets the current player
        '''
        return self._current_player

    def set_board(self, board):
        self._board = board

    def select(self, pieceIndex: int) -> bool:
        '''
        select a piece. Returns True on success
        '''
        if pieceIndex not in self._board:
            self.__selected_piece_index = pieceIndex
            return True
        return False

    def place(self, x: int, y: int) -> bool:
        '''
        Place piece in coordinates (x, y). Returns true on success
        '''
        if self.__placeable(x, y):
            self._board[y, x] = self.__selected_piece_index
            self._binary_board[y, x][:] = self.__pieces[self.__selected_piece_index].binary
            return True
        return False

    def __placeable(self, x: int, y: int) -> bool:
        return not (y < 0 or x < 0 or x > 3 or y > 3 or self._board[y, x] >= 0)

    def print(self):
        '''
        Print the board
        '''
        for row in self._board:
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
        return copy.deepcopy(self._board)

    def get_selected_piece(self) -> int:
        '''
        Get index of selected piece
        '''
        return copy.deepcopy(self.__selected_piece_index)

    def __check_horizontal(self) -> int:
        hsum = np.sum(self._binary_board, axis=1)

        if self.BOARD_SIDE in hsum or 0 in hsum:
            return self._current_player
        else:
            return -1

    def __check_vertical(self):
        vsum = np.sum(self._binary_board, axis=0)

        if self.BOARD_SIDE in vsum or 0 in vsum:
            return self._current_player
        else:
            return -1

    def __check_diagonal(self):
        dsum1 = np.trace(self._binary_board, axis1=0, axis2=1)
        dsum2 = np.trace(np.fliplr(self._binary_board), axis1=0, axis2=1)

        if self.BOARD_SIDE in dsum1 or self.BOARD_SIDE in dsum2 or 0 in dsum1 or 0 in dsum2:
            return self._current_player
        else:
            return -1

    def check_winner(self) -> int:
        '''
        Check who is the winner
        '''
        l = [self.__check_horizontal(), self.__check_vertical(),
             self.__check_diagonal()]
        for elem in l:
            if elem >= 0:
                return elem
        return -1

    def check_finished(self) -> bool:
        '''
        Check who is the loser
        '''
        for row in self._board:
            for elem in row:
                if elem == -1:
                    return False
        return True

    # def run(self) -> int:
    #     '''
    #     Run the game (with output for every move)
    #     '''
    #     winner = -1
    #     while winner < 0 and not self.check_finished():
    #         logging.info(f"Player {self._current_player} turn")
    #         # self.print()
    #         piece_ok = False
    #         while not piece_ok:
    #             piece_ok = self.select(self.__players[self._current_player].choose_piece(
    #                 self._board, self.__selected_piece_index))
    #         piece_ok = False
    #         self._current_player = (
    #             self._current_player + 1) % self.MAX_PLAYERS
    #         # self.print()
    #         while not piece_ok:
    #             x, y = self.__players[self._current_player].place_piece(
    #                 self._board, self.__selected_piece_index)
    #             piece_ok = self.place(x, y)
    #         # print(self.state())
    #         winner = self.check_winner()
    #     # self.print()
    #     return winner
    def run(self) -> int:
        '''
        Run the game (with output for every move)
        '''
        winner = -1
        while winner < 0 and not self.check_finished():
            print(f"Player {self._current_player} turn")
            self.print()
            piece_ok = False
            while not piece_ok:
                piece_ok = self.select(
                    self.__players[self._current_player].choose_piece())
            piece_ok = False
            self._current_player = (
                self._current_player + 1) % self.MAX_PLAYERS
            self.print()
            while not piece_ok:
                x, y = self.__players[self._current_player].place_piece()
                piece_ok = self.place(x, y)
            winner = self.check_winner()
        self.print()
        return winner


class Quarto(QuartoParent):

    MAX_PLAYERS = 2
    BOARD_SIDE = 4
    MAX_PIECES = 16

    def __init__(self, pieces=None) -> None:
        super().__init__(pieces)

    def set_board(self, board):
        self._board = board

    def get_players(self):
        return super().get_players()

    def set_selected_piece(self, index: int):
        '''
        Set index of selected piece
        '''
        super().select(index)

    def check_is_game_over(self) -> bool:
        '''
        Check if the game is over
        '''
        logging.debug("Board: ", self._board)
        logging.debug("Winner: ", self.check_winner())
        logging.debug("Finished: ", self.check_finished())
        logging.debug("Draw: ", self.check_if_draw())
        return self.check_winner() >= 0 or self.check_finished() or self.check_if_draw()

    # def run(self) -> int:
    #     '''
    #     Run the game (with output for every move)
    #     '''
    #     winner = -1
    #     while winner < 0 and not self.check_finished():
    #         self.print()
    #         piece_ok = False
    #         while not piece_ok:
    #             piece_ok = self.select(
    #                 self.__players[self._current_player].choose_piece())
    #         piece_ok = False
    #         self._current_player = (
    #             self._current_player + 1) % self.MAX_PLAYERS
    #         self.print()
    #         while not piece_ok:
    #             x, y = self.__players[self._current_player].place_piece()
    #             piece_ok = self.place(x, y)
    #         winner = self.check_winner()
    #     self.print()
    #     return winner

    def state(self) -> str:
        '''
        Return the state of the game
        '''
        return str(self._board)

    def state_as_array(self) -> np.ndarray:
        '''
        Return the state of the game as a numpy array
        '''
        return self._board

    def switch_player(self):
        '''
        Switch the current player
        '''
        self._current_player = 1 - self._current_player

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
        if piece in self._board.flatten():
            logging.debug("piece already in the board")
            return False
        time_end = datetime.now()

        # TODO: Put this back
        # if piece < 0 or piece >= self.MAX_PIECES:
        #     logging.debug("piece out of range")
        #     return False
        if x < 0 or x >= self.BOARD_SIDE:
            logging.debug("x out of range")
            return False
        if y < 0 or y >= self.BOARD_SIDE:
            logging.debug("y out of range")
            return False

        if self._board[y, x] > -1:
            logging.debug(f"position y, x: , {y}, {x} already occupied")
            # logging.debug(self._board)
            logging.debug("move to position already occupied")
            logging.debug(f"Index of -1: {np.where(self._board == -1)}")
            return False

        # move to position already occupied
        # if the next piece chosen is empty, then the move is invalid
        # if self._pieces[next_piece] == -1:
        #     return False
        # chosen piece already in the board
        if next_piece in self._board.flatten():
            logging.debug("chosen piece already in the board")
            return False

        num_empty_slots = 0
        for row in self._board.flatten():
            if row == -1:
                num_empty_slots += 1

        if piece == next_piece and not (num_empty_slots == 1):
            logging.debug("piece and next_piece are the same")
            return False

        return True

    def make_move(self, piece: int, x: int, y: int, next_piece: int, newboard=False, return_move=False):
        '''
        Make a move
        '''
        if newboard:
            new = copy.deepcopy(self)
            if new.check_if_move_valid(piece, x, y, next_piece):
                new._board[y, x] = piece
                new._binary_board[y, x][:] = new.get_piece_charachteristics(new.get_selected_piece()).binary

                new.set_selected_piece(next_piece)

                new._current_player = (
                    self._current_player + 1) % self.MAX_PLAYERS
            else:
                logging.debug("Invalid move")

            if return_move:
                return new, (piece, x, y, next_piece)
            return new
        else:
            if self.check_if_move_valid(piece, x, y, next_piece):
                # print("Turn: ", self._current_player)
                self._board[y, x] = piece
                self._binary_board[y, x][:] = self.get_piece_charachteristics(self.get_selected_piece()).binary
                self.set_selected_piece(next_piece)
                self._current_player = (
                    self._current_player + 1) % self.MAX_PLAYERS
            else:
                logging.debug("Invalid move")

            if return_move:
                return self, (piece, x, y, next_piece)
            return self

    def board_to_string(self):
        string = ''
        for row in self._board:
            for elem in row:
                string += str(elem) + ' '
        return string

    def string_to_board(self, string):
        board_elements = string.strip().split(' ')
        board = np.zeros((self.BOARD_SIDE, self.BOARD_SIDE))
        for i in range(len(board_elements)):
            board[i // self.BOARD_SIDE][i % self.BOARD_SIDE] = int(
                board_elements[i])
        # print(board)
        return board

    def state(self) -> str:
        '''
        Return the state of the game
        '''
        return str(self._board)

from abc import abstractmethod
import random
import quarto


class Player(object):
    '''
    Calabrese base class
    '''

    def __init__(self, quarto):
        self.__quarto = quarto

    @abstractmethod
    def choose_piece(self):
        pass

    @abstractmethod
    def place_piece(self):
        pass

    def get_game(self):
        return self.__quarto


class RandomPlayer(quarto.Player):
    """Random player"""

    def __init__(self, quarto: quarto.Quarto):
        super().__init__(quarto)

    def choose_piece(self, state=None, idk: int = None):
        return random.randint(0, 15)

    def place_piece(self, state=None, piece_to_be_placed: int = None):
        return random.randint(0, 3), random.randint(0, 3)


class HumanPlayer(quarto.Player):
    """Human player"""

    def __init__(self, quarto: quarto.Quarto):
        super().__init__(quarto)

    def choose_piece(self):
        return int(input("Choose piece: "))

    def place_piece(self):
        return int(input("Place piece: ")) // 4, int(input("Place piece: ")) % 4

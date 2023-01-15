import random
import quarto


class RandomPlayer(quarto.Player):
    """Random player"""

    def __init__(self, quarto: quarto.Quarto):
        super().__init__(quarto)

    def choose_piece(self, state=None):
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

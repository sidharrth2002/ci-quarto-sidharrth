# Free for personal or classroom use; see 'LICENSE.md' for details.
# https://github.com/squillero/computational-intelligence

import logging
import argparse
import random
import quarto

class RandomPlayer(quarto.Player):
    """Random player"""

    def __init__(self, quarto: quarto.Quarto):
        super().__init__(quarto)

    def choose_piece(self):
        return random.randint(0, 15)

    def place_piece(self):
        return random.randint(0, 3), random.randint(0, 3)

class HumanPlayer(quarto.Player):
    """Human player"""

    def __init__(self, quarto: quarto.Quarto):
        super().__init__(quarto)

    def choose_piece(self):
        return int(input("Choose piece: "))

    def place_piece(self):
        return int(input("Place piece: ")) // 4, int(input("Place piece: ")) % 4

class ReinforcementPlayer(quarto.Player):
    """Reinforcement player"""

    def __init__(self, quarto: quarto.Quarto):
        super().__init__(quarto)
        self._q = {}
        self._alpha = 0.1
        self._gamma = 0.9
        self._epsilon = 0.1
        self._last_state = None
        self._last_action = None

    def choose_piece(self):
        """Choose a piece on the board"""
        state = self._quarto.state()
        if state not in self._q:
            self._q[state] = [0] * 16
        if random.random() < self._epsilon:
            action = random.randint(0, 15)
        else:
            action = self._q[state].index(max(self._q[state]))
        self._last_state = state
        self._last_action = action
        return action

    def place_piece(self):
        """Choose a place on the board to place the piece"""
        state = self._quarto.state()
        if state not in self._q:
            self._q[state] = [0] * 16
        if random.random() < self._epsilon:
            action = random.randint(0, 15)
        else:
            action = self._q[state].index(max(self._q[state]))
        self._q[self._last_state][self._last_action] += self._alpha * (
            self._quarto.score() + self._gamma * self._q[state][action] - self._q[self._last_state][self._last_action])
        return action // 4, action % 4

    def train(self, n_games=1000):
        """Train the player against random opponent"""
        for i in range(n_games):
            self._quarto.set_players((self, RandomPlayer(self._quarto)))
            self._quarto.run()
            logging.debug(f"train: game {i} done")

def main():
    game = quarto.Quarto()
    game.set_players((RandomPlayer(game), RandomPlayer(game)))
    winner = game.run()
    logging.warning(f"main: Winner: player {winner}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='count', default=0, help='increase log verbosity')
    parser.add_argument('-d',
                        '--debug',
                        action='store_const',
                        dest='verbose',
                        const=2,
                        help='log debug messages (same as -vv)')
    args = parser.parse_args()

    if args.verbose == 0:
        logging.getLogger().setLevel(level=logging.WARNING)
    elif args.verbose == 1:
        logging.getLogger().setLevel(level=logging.INFO)
    elif args.verbose == 2:
        logging.getLogger().setLevel(level=logging.DEBUG)

    main()
import logging
from GeneticAlgorithm.ga_score_board import FinalPlayer
from Hardcoded.hardcoded import HardcodedPlayer
from lib.players import RandomPlayer
from quarto.objects import Quarto


def main():
    wins = 0
    for game in range(10):
        game = Quarto()
        game.set_players((FinalPlayer(game), RandomPlayer(game)))
        winner = game.run()
        print(f"main: Winner: player {winner}")
        if winner == 0:
            wins += 1
    print(f"main: Wins: {wins}/10")

if __name__ == "__main__":
    main()
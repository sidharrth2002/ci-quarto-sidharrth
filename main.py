import logging
from GeneticAlgorithm.ga_score_board import FinalPlayer
from Hardcoded.hardcoded import HardcodedPlayer
from lib.players import RandomPlayer
from quarto.objects import Quarto


def main():
    wins = 0
    for game in range(10):
        game = Quarto()
        # 0 if plays first, 1 if plays second
        # IMPORTANT: this must be set to make sure the MCTS reward function works correctly
        agent_position = 1
        game.set_players((RandomPlayer(game), FinalPlayer(game, agent_position=agent_position)))
        winner = game.run()
        print(f"main: Winner: player {winner}")
        if winner == 1:
            wins += 1
    print(f"main: Wins: {wins}/10")

if __name__ == "__main__":
    main()
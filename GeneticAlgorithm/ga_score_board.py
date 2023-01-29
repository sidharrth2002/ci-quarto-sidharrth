'''
Genetic Algorithm for Quarto
'''
import sys
sys.path.insert(0, '..')

import tqdm
import random
import logging
import json
import itertools
from copy import deepcopy
from lib.players import Player, RandomPlayer
from quarto.objects import Quarto
from lib.scoring import score_board
from QLMCTS import QLearningPlayer
from Hardcoded.hardcoded import HardcodedPlayer

logging.basicConfig(level=logging.DEBUG)

class Genome:
    def __init__(self, thresholds, fitness):
        self.thresholds = thresholds
        self.fitness = fitness

    def set_fitness(self, fitness):
        self.fitness = fitness

    def set_thresholds(self, thresholds):
        self.thresholds = thresholds

    def toJSON(self):
        return {
            'thresholds': self.thresholds,
            'fitness': self.fitness
        }


class FinalPlayer(Player):
    '''
    Final player uses genetic algorithm to decide between:
    1. Hardcoded Strategy
    2. Random Strategy
    3. QL-MCTS
    '''

    def __init__(self, quarto: Quarto = None):
        if quarto is None:
            quarto = Quarto()
        super().__init__(quarto)
        self.ql_mcts = QLearningPlayer(quarto)
        self.hardcoded = HardcodedPlayer(quarto)
        self.random_player = RandomPlayer(quarto)
        self.BOARD_SIDE = 4
        self.GENOME_VAL_UPPER_BOUND = 16
        self.GENOME_VAL_LOWER_BOUND = 0
        self.thresholds = {
            'random': 1.090773081612301,
            'hardcoded': 2.790328881747581,
            'ql-mcts': 8.251997327518943
        }

    def generate_population(self, population_size):
        population = []
        for i in range(population_size):
            threshold = {}

            # make sure that value for random < hardcoded < ql-mcts
            threshold['random'] = random.random() * self.GENOME_VAL_UPPER_BOUND
            # find random number between random and 15
            threshold['hardcoded'] = threshold['random'] + \
                random.random() * (self.GENOME_VAL_UPPER_BOUND -
                                   threshold['random'])

            # find random number between hardcoded and 15
            threshold['ql-mcts'] = threshold['hardcoded'] + \
                random.random() * (self.GENOME_VAL_UPPER_BOUND -
                                   threshold['hardcoded'])

            assert threshold['random'] < threshold['hardcoded'] < threshold['ql-mcts']

            population.append(Genome(threshold, 0))
        return population

    def ensure_correct_ordering(self, new_thresholds):
        if new_thresholds['random'] > new_thresholds['hardcoded']:
            new_thresholds['random'], new_thresholds['hardcoded'] = new_thresholds['hardcoded'], new_thresholds['random']
        if new_thresholds['hardcoded'] > new_thresholds['ql-mcts']:
            new_thresholds['hardcoded'], new_thresholds['ql-mcts'] = new_thresholds['ql-mcts'], new_thresholds['hardcoded']
        if new_thresholds['random'] > new_thresholds['hardcoded']:
            new_thresholds['random'], new_thresholds['hardcoded'] = new_thresholds['hardcoded'], new_thresholds['random']
        return new_thresholds

    def crossover(self, genome1, genome2):
        new_thresholds = {}
        for key in genome1.thresholds:
            new_thresholds[key] = random.choice(
                [genome1.thresholds[key], genome2.thresholds[key]])

        # make sure that value for random < hardcoded < ql-mcts
        new_thresholds = self.ensure_correct_ordering(new_thresholds)
        return Genome(new_thresholds, 0)

    def mutate(self, genome):
        new_thresholds = {}
        genome_thresholds = genome.thresholds
        if random.random() < 0.4:
            new_thresholds['random'] = random.random() * \
                self.GENOME_VAL_UPPER_BOUND
            new_thresholds['hardcoded'] = random.choice(
                [genome_thresholds['random'], genome_thresholds['random'] +
                 random.random() * (self.GENOME_VAL_UPPER_BOUND - genome_thresholds['random'])])
            new_thresholds['ql-mcts'] = random.choice(
                [genome_thresholds['hardcoded'], genome_thresholds['hardcoded'] +
                 random.random() * (self.GENOME_VAL_UPPER_BOUND - genome_thresholds['hardcoded'])])

            new_thresholds = self.ensure_correct_ordering(new_thresholds)

            assert new_thresholds['random'] < new_thresholds['hardcoded'] < new_thresholds['ql-mcts']

            return Genome(new_thresholds, 0)
        return genome

    def evolve(self, num_generations=50):
        self.population_size = 50
        self.offspring_size = 10
        population = self.generate_population(self.population_size)

        pbar = tqdm.tqdm(total=num_generations)
        for gen in range(num_generations):
            pbar.update(1)
            logging.debug('Generation: {}'.format(gen))
            offpsring = []
            for i in range(self.offspring_size):
                parent1 = random.choice(population)
                parent2 = random.choice(population)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                child.fitness = self.play_game(child.thresholds, num_games=5)
                offpsring.append(child)
            population += offpsring
            population = sorted(
                population, key=lambda x: x.fitness, reverse=True)[:self.population_size]

            if gen % 5 == 0:
                logging.info('Saving population')
                with open('/Volumes/USB/population3.json', 'w') as f:
                    json.dump([genome.toJSON() for genome in population], f)

        # return the best genome's thresholds
        return population[0].thresholds

    def play_game(self, thresholds, num_games=10):
        wins = 0
        for game in range(num_games):
            logging.debug('Game: {}'.format(game))
            state = Quarto()
            player = 0

            # initialise with some random piece just to kickstart game
            state.set_selected_piece(self.random_player.choose_piece(state, 0))
            self.current_state = state

            # python passes by reference
            # agent will use the state, etc. to update the Q-table
            # this function also wipes the MCTS tree
            self.ql_mcts.clear_and_set_current_state(state)
            self.hardcoded = HardcodedPlayer(state)

            while True:
                # board score is the number of couples and triplets on the board
                # it is indicative of the change of the board state
                board_score = score_board(self.current_state)

                differences = [abs(board_score - thresholds[key])
                               for key in thresholds]
                min_diff = min(differences)
                index = differences.index(min_diff)
                key = list(thresholds.keys())[index]

                if player == 0:
                    if key == 'random':
                        logging.debug('random')
                        # play randomly
                        action = self.random_player.place_piece()
                        next_piece = self.random_player.choose_piece()
                        while self.current_state.check_if_move_valid(self.current_state.get_selected_piece(), action[0], action[1], next_piece) is False:
                            action = self.random_player.place_piece()
                            next_piece = self.random_player.choose_piece()
                        self.current_state.select(
                            self.current_state.get_selected_piece())
                        self.current_state.place(action[0], action[1])
                        self.current_state.set_selected_piece(next_piece)
                        self.current_state.switch_player()
                        player = 1 - player

                    elif key == 'hardcoded':
                        # play using hardcoded strategy
                        logging.debug('hardcoded')
                        self.previous_state = deepcopy(self.current_state)
                        winning_piece, position = self.hardcoded.hardcoded_strategy_get_move(
                            self.current_state)
                        next_piece = self.hardcoded.hardcoded_strategy_get_piece(
                            self.current_state)
                        while self.current_state.check_if_move_valid(self.current_state.get_selected_piece(), position[0], position[1], next_piece) is False:
                            winning_piece, position = self.hardcoded.hardcoded_strategy_get_move(
                                self.current_state)
                            next_piece = self.hardcoded.hardcoded_strategy_get_piece(
                                self.current_state)
                        self.current_state.select(state.get_selected_piece())
                        self.current_state.place(position[0], position[1])
                        self.current_state.set_selected_piece(next_piece)
                        self.current_state.switch_player()
                        player = 1 - player

                    else:
                        # play using QL-MCTS
                        logging.debug('ql-mcts')
                        self.ql_mcts.previous_state = deepcopy(
                            self.current_state)
                        action = self.ql_mcts.get_action(self.current_state)
                        self.ql_mcts.previous_action = action
                        self.ql_mcts.current_state.select(
                            self.current_state.get_selected_piece())
                        self.ql_mcts.current_state.place(action[0], action[1])
                        self.ql_mcts.current_state.set_selected_piece(
                            action[2])
                        self.ql_mcts.current_state.switch_player()
                        player = 1 - player

                else:
                    # opponent is random
                    action = self.random_player.place_piece()
                    next_piece = self.random_player.choose_piece()
                    while self.current_state.check_if_move_valid(self.current_state.get_selected_piece(), action[0], action[1], next_piece) is False:
                        action = self.random_player.place_piece()
                        next_piece = self.random_player.choose_piece()
                        # WARNING: very often stuck in this loop
                    self.current_state.select(
                        self.current_state.get_selected_piece())
                    self.current_state.place(action[0], action[1])
                    self.current_state.set_selected_piece(next_piece)
                    self.current_state.switch_player()
                    player = 1 - player

                if self.current_state.check_is_game_over():
                    if 1 - self.current_state.check_winner() == 0:
                        print("Agent wins")
                        wins += 1
                        # TODO: QL reward update
                    else:
                        print("Player 2 wins")
                    break

        # fitness is the percentage of games won
        logging.debug(f"Win rate: {wins/num_games}")
        return wins/num_games

    def choose_piece(self):
        '''
        Choose piece for next player to place
        '''
        thresholds = self.thresholds

        # game is stored in parent
        self.current_state = self.get_game()

        board_score = score_board(self.current_state)

        differences = [abs(board_score - thresholds[key])
                       for key in thresholds]
        min_diff = min(differences)
        index = differences.index(min_diff)
        key = list(thresholds.keys())[index]

        if key == 'random':
            logging.debug('random')
            # play randomly
            action = self.random_player.place_piece()
            next_piece = self.random_player.choose_piece()
            while self.current_state.check_if_move_valid(self.current_state.get_selected_piece(), action[0], action[1], next_piece) is False:
                action = self.random_player.place_piece()
                next_piece = self.random_player.choose_piece()
            return next_piece

        elif key == 'hardcoded':
            # play using hardcoded strategy
            logging.debug('hardcoded')
            self.previous_state = deepcopy(self.current_state)
            winning_piece, position = self.hardcoded_strategy_get_move(
                self.current_state)
            next_piece = self.hardcoded_strategy_get_piece(
                self.current_state)
            while self.current_state.check_if_move_valid(self.current_state.get_selected_piece(), position[0], position[1], next_piece) is False:
                winning_piece, position = self.hardcoded_strategy_get_move(
                    self.current_state)
                next_piece = self.hardcoded_strategy_get_piece(
                    self.current_state)
            return next_piece

        else:
            # play using QL-MCTS
            logging.debug('ql-mcts')
            self.ql_mcts.previous_state = deepcopy(
                self.current_state)
            action = self.ql_mcts.get_action(self.current_state)
            self.ql_mcts.previous_action = action
            self.ql_mcts.current_state.select(
                self.current_state.get_selected_piece())
            return action[2]

    def place_piece(self):
        # python passes by reference
        # agent will use the state, etc. to update the Q-table
        # this function also wipes the MCTS tree
        self.current_state = self.get_game()
        thresholds = self.thresholds

        self.ql_mcts.clear_and_set_current_state(self.current_state)

        while True:
            # board score is the number of couples and triplets on the board
            # it is indicative of the change of the board state
            board_score = score_board(self.current_state)

            differences = [abs(board_score - thresholds[key])
                           for key in thresholds]
            min_diff = min(differences)
            index = differences.index(min_diff)
            key = list(thresholds.keys())[index]

            if key == 'random':
                logging.debug('random')
                # play randomly
                action = self.random_player.place_piece()
                next_piece = self.random_player.choose_piece()
                while self.current_state.check_if_move_valid(self.current_state.get_selected_piece(), action[0], action[1], next_piece) is False:
                    action = self.random_player.place_piece()
                    next_piece = self.random_player.choose_piece()
                return action[0], action[1]

            elif key == 'hardcoded':
                # play using hardcoded strategy
                logging.debug('hardcoded')
                self.previous_state = deepcopy(self.current_state)
                winning_piece, position = self.hardcoded_strategy_get_move(
                    self.current_state)
                next_piece = self.hardcoded_strategy_get_piece(
                    self.current_state)
                while self.current_state.check_if_move_valid(self.current_state.get_selected_piece(), position[0], position[1], next_piece) is False:
                    winning_piece, position = self.hardcoded_strategy_get_move(
                        self.current_state)
                    next_piece = self.hardcoded_strategy_get_piece(
                        self.current_state)
                return position[0], position[1]

            else:
                # play using QL-MCTS
                logging.debug('ql-mcts')
                self.ql_mcts.previous_state = deepcopy(
                    self.current_state)
                action = self.ql_mcts.get_action(self.current_state)
                self.ql_mcts.previous_action = action
                return action[0], action[1]

    def test_thresholds(self):
        thresholds = {'random': 10000,
                      'hardcoded': 10000, 'ql-mcts': 3}
        win_rate = self.play_game(thresholds, num_games=10)
        return win_rate

if __name__ == "__main__":
    final_player = FinalPlayer()
    # best_thresholds = final_player.evolve()
    # print(best_thresholds)

    average_win_rate = 0
    for i in range(10):
        win_rate = final_player.test_thresholds()
        average_win_rate += win_rate
    print(f"Average win rate: {average_win_rate/10}")
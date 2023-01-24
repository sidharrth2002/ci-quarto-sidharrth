'''
Genetic Algorithm for Quarto
'''
from copy import deepcopy
import itertools
import json
import logging
import random
from QLearningPlayer import QLearningPlayer

from lib.players import Player, RandomPlayer
from quarto.objects import Quarto

logging.basicConfig(level=logging.INFO)


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

    def __init__(self):
        quarto = Quarto()
        super().__init__(quarto)
        self.ql_mcts = QLearningPlayer(quarto)
        self.random_player = RandomPlayer(quarto)

    def check_if_winning_piece(self, state, piece):
        for i in range(self.BOARD_SIDE):
            for j in range(self.BOARD_SIDE):
                if state.check_if_move_valid(piece, i, j, piece):
                    cloned_state = deepcopy(state)
                    cloned_state.select(piece)
                    cloned_state.place(i, j)

                    if cloned_state.check_is_game_over():
                        return True, [i, j]
        return False, None

    def hardcoded_strategy_get_piece(self, state):
        possible_pieces = []
        for i in range(16):
            # check if the piece is a winning piece
            winning_piece, _ = self.check_if_winning_piece(state, i)
            if not winning_piece and i not in itertools.chain.from_iterable(state.state_as_array()):
                possible_pieces.append(i)

        return random.choice(possible_pieces)

    def hardcoded_strategy_get_move(self, state):
        #  1. Play the piece handed over by the opponent:
        # (a) play a winning position if handed a winning piece;
        # (b) otherwise, play to build a line of like pieces if possible;
        # (c) otherwise, play randomly.
        # 2. Hand a piece to the opponent:
        # (a) avoid handing over a winning piece for your opponent to play;
        # (b) otherwise, choose randomly.

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
                    return True, list(reversed(empty_rows[-1]))

        # play randomly
        for i in range(self.BOARD_SIDE):
            for j in range(self.BOARD_SIDE):
                for next_piece in range(16):
                    if state.check_if_move_valid(selected_piece, i, j, next_piece):
                        return False, [i, j]

    def generate_population(self, population_size):
        population = []
        for i in range(population_size):
            threshold = {
                'random': random.random(),
                'hardcoded': random.random(),
                'ql-mcts': random.random()
            }
            population.append(Genome(threshold, 0))
        return population

    def crossover(self, genome1, genome2):
        new_thresholds = {}
        for key in genome1.thresholds:
            new_thresholds[key] = random.choice(
                [genome1.thresholds[key], genome2.thresholds[key]])
        return Genome(new_thresholds, 0)

    def mutate(self, genome):
        new_thresholds = {}
        if random.random() < 0.4:
            for key in genome.thresholds:
                new_thresholds[key] = random.choice(
                    [genome.thresholds[key], random.random()])
            return Genome(new_thresholds, 0)
        return genome

    def evolve(self):
        self.population_size = 70
        self.offspring_size = 40
        population = self.generate_population(self.population_size)

        for gen in range(100):
            logging.info('Generation: {}'.format(gen))
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
                with open('/Volume/USB/population.json', 'w') as f:
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

            while True:
                # num pieces are the number of places on the board that are occupied
                num_pieces_in_board = 16 - list(itertools.chain.from_iterable(
                    state.state_as_array())).count(-1)
                if player == 0:
                    if num_pieces_in_board < thresholds['random']:
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

                    elif num_pieces_in_board > thresholds['random'] and num_pieces_in_board < thresholds['hardcoded']:
                        # play using hardcoded strategy
                        self.previous_state = deepcopy(self.current_state)
                        winning_piece, position = self.hardcoded_strategy_get_move(
                            self.current_state)
                        self.current_state.select(state.get_selected_piece())
                        self.current_state.place(position[0], position[1])
                        next_piece = self.hardcoded_strategy_get_piece(
                            self.current_state)
                        self.current_state.set_selected_piece(next_piece)
                        self.current_state.switch_player()
                    else:
                        # play using QL-MCTS
                        # time to go pro
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
                    self.current_state.select(
                        self.current_state.get_selected_piece())
                    self.current_state.place(action[0], action[1])
                    self.current_state.set_selected_piece(next_piece)
                    self.current_state.switch_player()
                    player = 1 - player

                if self.current_state.check_is_game_over():
                    if 1 - self.current_state.check_winner() == 0:
                        logging.debug("Player 1 wins")
                        wins += 1
                        # TODO: QL reward update
                    else:
                        logging.debug("Player 2 wins")
                    break

        # fitness is the percentage of games won
        logging.debug(f"Win rate: {wins/num_games}")
        return wins/num_games


final_player = FinalPlayer()
best_thresholds = final_player.evolve()
print(best_thresholds)

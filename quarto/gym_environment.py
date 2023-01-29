"""
In this file, I create 2 custom OpenAI Gym environments for Quarto.
"""

import itertools
import logging
import gym
from gym import spaces
import numpy as np
from lib.players import RandomPlayer

from quarto.objects import Quarto

class QuartoScapeNew(gym.Env):
    '''Custom gym environment for Quarto'''

    def __init__(self):
        self.game = Quarto()
        self.action_space = spaces.MultiDiscrete([16, 16, 16])
        self.observation_space = spaces.MultiDiscrete([17] * 17)
        self.reward_range = (-1, 1)
        self.main_player = None

    def set_main_player(self, player):
        self.main_player = player
        self.game.set_players((player, RandomPlayer(self.game)))
        return True

    def step(self, action, chosen_piece):
        # position is the position the previous piece should be moved to
        # chosen next piece is the piece the agent chooses for the next player to move
        x, y, chosen_next_piece = action
        self.next_piece = chosen_next_piece
        if self.game.check_if_move_valid(chosen_piece, x, y, chosen_next_piece):
            # print(f"Valid move, piece {chosen_piece} placed at {x}, {y}")
            self.game.select(chosen_piece)
            self.game.place(x, y)
            # self.game.print()
            if self.game.check_is_game_over():
                # just playing with itself
                logging.info("Giving reward of 1 for completing the game")
                reward = 1
                return self.game.state_as_array(), self.game.check_winner(), self.game.check_finished(), {}
            else:
                logging.info("Giving reward of 0 for making a move that didn't end the game")
                reward = 0
                return self.game.state_as_array(), self.game.check_winner(), self.game.check_finished(), {}

        else:
            reward = -1

        return self.game.state_as_array(), reward, self.game.check_finished(), {}

    def reset(self):
        self.game = Quarto()
        self.game.set_players((self.main_player, RandomPlayer(self.game)))
        # print(self.game.state_as_array())
        return self.game.state_as_array()




class QuartoScape(gym.Env):
    '''Custom gym environment for Quarto'''

    def __init__(self):
        self.game = Quarto()
        self.action_space = spaces.MultiDiscrete([4, 4, 16])
        self.observation_space = spaces.MultiDiscrete([18] * 17)
        self.reward_range = (-1, 1)
        self.main_player = None

    def close(self):
        pass

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
        # print("Placing in reward")
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

    def step(self, action, chosen_piece):
        # position is the position the previous piece should be moved to
        # chosen next piece is the piece the agent chooses for the next player to move
        x, y, chosen_next_piece = action
        self.next_piece = chosen_next_piece
        reward = 0
        if self.game.check_if_move_valid(chosen_piece, x, y, chosen_next_piece):
            logging.info(
                f"Valid move, piece {chosen_piece} placed at {x}, {y}")
            self.game.select(chosen_piece)
            # print(f"Trying to place piece {chosen_piece} at {x}, {y}")
            self.game.place(x, y)
            # self.game.print()
            if self.game.check_winner() != -1:
                # this move resulted in a win
                reward = 100
                return self.game.state_as_array(), reward, self.game.check_is_game_over(), {}
            else:
            #     # this move did not result in a win or a draw
                reward = self.reward(
                    self.game.state_as_array(), chosen_piece, action)
                # reward = 0
                return self.game.state_as_array(), reward, self.game.check_is_game_over(), {}
        return self.game.state_as_array(), reward, self.game.check_finished(), {}

    def reset(self):
        self.game = Quarto()
        self.game.set_players((self.main_player, RandomPlayer(self.game)))
        arr = np.array(list(itertools.chain.from_iterable(
            self.game.state_as_array())) + [100])
        # replace -1 with 17
        arr[arr == -1] = 17
        return arr

    def close(self):
        logging.info("Closing environment")
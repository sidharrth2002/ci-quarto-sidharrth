from copy import deepcopy
import hashlib
import itertools
import os
import random
import numpy as np
from lib.isomorphic import BoardTransforms
from quarto.objects import Quarto


class Node:
    def __init__(self, state: Quarto = Quarto(), place_current_move=None, final_point=False):
        self._state = state
        self.place_current_move = place_current_move
        self.final_point = final_point
        self.wins = 0
        self.visits = 0

    def __hash__(self):
        string = str(self._state.get_selected_piece()) + np.array2string(self._state.state_as_array())
        return int(hashlib.sha1(string.encode('utf-8')).hexdigest(), 32)

    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return np.array_equal(self._state.state_as_array(), other._state.state_as_array()) and self._state.get_selected_piece() == other._state.get_selected_piece()

    def child_already_exists(self, new_state: Quarto):
        board_new_state = new_state.state_as_array()
        for child in self._children:
            if BoardTransforms.compare_boards(board_new_state, child._state.state_as_array()):
                return True

        return False

    def update(self, reward: int):
        self.visits += 1
        self.wins += reward

    def reward(self, player_id):
        player_last_moved = 1 - self._state.get_current_player()

        player_who_last_moved = 1 - self._state.get_current_player()

        # 0 if plays first, 1 if plays second
        agent_position = player_id

        if player_who_last_moved == agent_position and 1 - self._state.check_winner() == agent_position:
            # MCTS won
            return 1
        elif player_who_last_moved == 1 - agent_position and 1 - self._state.check_winner() == 1 - agent_position:
            # MCTS lost
            return 0
        elif self._state.check_winner() == -1:
            # Draw game
            return 0.5

    def find_random_child(self):
        free_positions = []
        board = self._state.state_as_array()
        for i in range(4):
            for j in range(4):
                if board[i][j] == -1:
                    free_positions.append((i, j))
        place = random.choice(free_positions)
        new_quarto = deepcopy(self._state)
        # new_quarto = Quarto(board=self._state.state_as_array(), selected_piece=self._state.get_selected_piece(), curr_player=self._state.get_current_player())
        new_quarto.place(place[1], place[0])
        if new_quarto.check_finished() or new_quarto.check_winner() != -1:
            final_point = True
        else:
            new_board = list(itertools.chain.from_iterable(new_quarto.state_as_array()))
            free_pieces = [piece for piece in range(0, 16) if piece not in new_board]
            piece = random.choice(free_pieces)
            new_quarto.select(piece)
            final_point = False
        new_quarto._current_player = 1 - new_quarto._current_player
        return Node(new_quarto, place, final_point)


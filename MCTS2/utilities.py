import copy
import random
import numpy as np
from quarto.objects import Quarto


class Node:
    def __init__(self, state: Quarto, place_current_move=None, final_point=False):
        self.state = state
        self.place_current_move = place_current_move
        self.final_point = final_point
        self.wins = 0
        self.visits = 0

    def __hash__(self):
        return str(self.state.get_selected_piece()) + np.array2string(self.state.state_as_array())

    def child_already_exists(self, new_state: Quarto):
        board_new_state = new_state.state_as_array()
        rotate_90_clockwise, rotate_90_counter_clockwise, reflect_horizontal, reflect_vertical = self._functions.symmetries(
            board_new_state)
        for child in self._children:
            board_already_present = child._state.get_board_status()
            if (np.array_equal(board_new_state, board_already_present) or
                np.array_equal(rotate_90_clockwise, board_already_present) or
                np.array_equal(rotate_90_counter_clockwise, board_already_present) or
                np.array_equal(reflect_horizontal, board_already_present) or
                np.array_equal(reflect_vertical, board_already_present)):
                    return True

        return False

    def update(self, reward: int):
        self.visits += 1
        self.wins += reward

    def reward(self, player_id):
        player_last_moved = 1 - self.state.get_current_player()

        player_who_last_moved = 1 - self.state.get_current_player()

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
        board = self.state.state_as_array()
        for i in range(4):
            for j in range(4):
                if board[i][j] == -1:
                    free_positions.append((i, j))
        place = random.choice(free_positions)
        new_quarto = copy.deepcopy(self.state)
        new_quarto.place(place[1], place[0])
        if new_quarto.check_finished() or new_quarto.check_winner() != -1:
            end_point = True
        else:
            new_board = new_quarto.state_as_array()
            free_pieces = [piece for piece in range(0, 16) if piece not in board]
            piece = random.choice(free_pieces)
            new_quarto.select(piece)
            end_point = False
        new_quarto._current_player = 1 - new_quarto._current_player
        return Node(new_quarto, place, end_point)


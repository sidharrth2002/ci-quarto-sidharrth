from collections import defaultdict
import math
import random


class QLearningPlayer:
    def __init__(self, board, epsilon=0.1, alpha=0.5, gamma=0.9):
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.board = board
        self.MAX_PIECES = 16
        self.BOARD_SIDE = 4
        self.Q = defaultdict(int)

    def hash_state_action(self, state, action):
        return hash(str(state) + '|' + str(action))

    def get_Q(self, state, action):
        if self.hash_state_action(state, action) not in self.Q:
            return None
        return self.Q[self.hash_state_action(state, action)]

    def get_Q_for_state(self, state):
        if self.hash_state_action(state, None) not in self.Q:
            return None
        return [i for i in self.Q if i.startswith(str(state))]

    def set_Q(self, state, action, value):
        self.Q[self.hash_state_action(state, action)] = value

    def get_possible_actions(self, state):
        actions = []
        for i in range(self.BOARD_SIDE):
            for j in range(self.BOARD_SIDE):
                for piece in state.get_available_pieces():
                    if state.check_if_move_valid(state.get_selected_piece(), i, j, piece):
                        actions.append((i, j, piece))
        return actions

    def get_max_Q(self, state):
        max_Q = -math.inf
        for action in self.get_possible_actions(state):
            max_Q = max(max_Q, self.get_Q(state, action))
        return max_Q

    def get_action(self, state):
        '''
        If state, action pair not in Q, go to Monte Carlo Tree Search to find best action
        '''
        if random.random() < self.epsilon:
            return random.choice(self.get_possible_actions(state))
        else:
            expected_score = 0
            for action in self.get_possible_actions(state):
                if self.get_Q(state, action) is not None and expected_score < self.get_Q(state, action):
                    expected_score = self.get_Q(state, action)
                    best_action = action
            # go to Monte Carlo Tree Search if no suitable action found in Q table
            if best_action is None:
                # print("Monte Carlo Tree Search")
                best_action = tree.choose(state)
            return best_action

    def update_Q(self, state, action, reward, next_state):
        self.set_Q(state, action, self.get_Q(state, action) + self.alpha *
                   (reward + self.gamma * self.get_max_Q(next_state) - self.get_Q(state, action)))

    def train(self):
        # 1. Select an action using Monte Carlo Tree Search
        # 2. Make the move
        # 3. Generate future probabilities using Monte Carlo Tree Search
        # 4. Dot product of future probabilities and Q values to get expected values
        # 5. Update Q table with formula: (1 - alpha) * Q[state, action] + alpha * (reward + gamma * np.max(expected_values))
        # 6. Repeat until game over

        while True:
            new_board = tree.choose(board)
            if Node(new_board).is_terminal():
                done = True
            future_probabilities = tree.generate_future_probabilities(
                new_board)

            # dot product between 2 dictionaries based on keys
            expected_values = {}
            for action, probability in future_probabilities.items():
                expected_values[action] = probability * self.Q[self.hash_state_action(
                    new_board, action)]

            # expected_values = np.dot(future_probabilities,
            #                          self.get_Q_for_state(new_board))
            self.Q[self.hash_state_action(board, action)] = (1 - self.alpha) * self.Q[self.hash_state_action(board, action)] + self.alpha * (
                reward + self.gamma * np.max(expected_values.values()))

            if done:
                break

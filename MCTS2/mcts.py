'''
In this file, we build an MCTS player using a different, simpler node structure.
'''

import copy
import itertools
import logging
import math
import random
from lib.players import Player
from quarto.objects import Quarto
from .node import Node


class MCTS(Player):
    def __init__(self, board, player_id = 0):
        '''
        Initialise player with empty children dictionary
        and player id (indicates position MCTS plays)
        This is important for reward function.
        '''
        # by default MCTS is player 0
        self.children = dict()
        self._player_id = player_id
        print(f"Player {self._player_id} initialised")
        super().__init__(board)

    def set_board(self, board):
        self._board = board

    def uct(self, node, child):
        '''
        Apply UCT formula to select best child
        Formula: UCT = wins/visits + sqrt(2*log(parent_visits)/child_visits)
        '''
        return child._wins/child._visits + math.sqrt(2*math.log(node._visits)/child._visits)

    def select(self, node: Node):
        '''
        Select the child with the highest UCT value
        '''
        points = []
        for child in self.children[node]:
            points.append((child, self.uct(node, child)))

        return max(points, key=lambda x: x[1])[0]

    def traverse(self, node: Node):
        '''
        Traverse the tree to find the leaf node
        '''
        path = []
        while True:
            path.append(node)
            if node not in self.children or not not self.children[node]:
                return path

            unexplored = self.children[node] - self.children.keys()
            if unexplored:
                path.append(unexplored.pop())
                return path
            node = self.select(node)

    def expand(self, node: Node):
        '''
        Expands from the leaf node to a state that is hopefully terminal. In this approach (different from MCTS1), the next piece is not passed down to the next node, but is directly applied to all empty positions.
        '''
        if node.final_point:
            self.children[node] = None
            return

        free_places = []
        board = node._state.state_as_array()
        for i in range(4):
            for j in range(4):
                if board[i][j] == -1:
                    free_places.append((i, j))

        children = []
        for y, x in free_places:
            quarto = copy.deepcopy(node._state)
            quarto.place(x, y)
            if quarto.check_finished() or quarto.check_winner() != -1:
                n = Node(copy.deepcopy(quarto), (x, y), True)
                children.append(n)
            else:
                free_pieces = [i for i in range(16) if i not in list(
                    itertools.chain.from_iterable(quarto.state_as_array()))]
                for piece in free_pieces:
                    new_quarto = copy.deepcopy(quarto)
                    new_quarto.select(piece)
                    new_quarto._current_player = (
                        new_quarto._current_player + 1) % 2
                    child = Node(new_quarto, (x, y))
                    children.append(child)
        self.children[node] = children

    def simulate(self, node: Node):
        '''
        Simulate until terminal state is reached
        '''
        while True:
            if node.final_point:
                reward = node.reward(self._player_id)
                return reward
            node = node.find_random_child()

    def backpropagate(self, reward, path):
        '''
        Backpropagate reward to all nodes in path
        (Invert rewards based on player id)
        '''
        for node in reversed(path):
            node.update(reward)
            reward = 1 - reward

    def best_child(self, node: Node):
        '''
        Choose best child purely based on wins and visits
        '''
        if node.final_point:
            raise RuntimeError(f'called on unterminal node')

        def score(n):
            logging.debug(f"Before reading in choose {n}")
            if n.visits == 0:
                return float('-inf')
            return self.wins[n] / self.visits[n]

        return max(self.children[node], key=score)

    def search(self, node: Node):
        '''
        1. Traverse tree to find leaf node
        2. Expand leaf node
        3. Simulate from leaf node until terminal state is reached
        4. Backpropagate reward to all nodes in path
        '''
        path = self.traverse(node)
        leaf = path[-1]
        self.expand(leaf)
        reward = self.simulate(leaf)
        self.backpropagate(reward, path)

    def do_rollout(self, root: Quarto):
        '''
        Create node and rollout from it
        '''
        if type(root) != Node:
            root = Node(state=root)
        self.search(root)
        return self.best_child(root)

    def choose_piece(self):
        '''
        Subclassed from Calabrese's player class. Will return a random piece if first move. If not, will return piece computed in `place_piece`
        '''
        if self.mcts_last_board == None:
            return random.randint(0, 15)
        else:
            return self.mcts_last_board._state.get_selected_piece()

    def place_piece(self):
        '''
        Iterate through and rollout before returning best child (next move to make)
        Since parent player class expects position and next piece to be
        returned by separate functions, next piece is stored in a variable in order to be called by `choose_piece`
        '''
        board = self.get_game().state_as_array()
        selected_piece = self.get_game().get_selected_piece()
        curr_player = self.get_game().get_current_player()
        current_board = Quarto(
            board=board, selected_piece=selected_piece, curr_player=curr_player)
        root = Node(current_board)
        self._player_id = self.get_game().get_current_player()
        for _ in range(30):
            best_child = self.do_rollout(root)
        self.mcts_last_board = best_child
        return best_child.place_current_move

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
        # by default MCTS is player 0
        self.children = dict()
        self._player_id = player_id
        super().__init__(board)

    def select(self, node: Node):
        points = []
        for child in self.children[node]:
            points.append((child, child._wins/child._visits +
                          math.sqrt(2*math.log(node._visits)/child._visits)))

        return max(points, key=lambda x: x[1])[0]

    def traverse(self, node: Node):
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
        while True:
            if node.final_point:
                reward = node.reward(self._player_id)
                return reward
            node = node.find_random_child()

    def backpropagate(self, reward, path):
        for node in reversed(path):
            node.update(reward)
            reward = 1 - reward

    def best_child(self, node: Node):
        if node.final_point:
            raise RuntimeError(f'called on unterminal node')

        def score(n):
            logging.debug(f"Before reading in choose {n}")
            if n.visits == 0:
                return float('-inf')
            return self.wins[n] / self.visits[n]

        return max(self.children[node], key=score)

    def search(self, node: Node):
        path = self.traverse(node)
        leaf = path[-1]
        self.expand(leaf)
        reward = self.simulate(leaf)
        self.backpropagate(reward, path)

    def do_rollout(self, root: Quarto):
        if type(root) != Node:
            root = Node(state=root)
        self.search(root)
        return self.best_child(root)

    def choose_piece(self):
        if self.mcts_last_board == None:
            return random.randint(0, 15)
        else:
            return self.mcts_last_board._state.get_selected_piece()

    def place_piece(self):
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

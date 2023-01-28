import math


class MinMaxPlayer:
    '''
    Min Max Player
    '''

    def __init__(self, depth=4):
        self.depth = depth

    def minmax(self, node, depth, maximizingPlayer, alpha, beta):
        if depth == 0 or node.is_terminal():
            if not node.is_terminal():
                return 0
            return node.reward()
        if maximizingPlayer:
            v = -math.inf
            for n in node.find_children():
                v = max(v, self.minmax(n, depth - 1, False, alpha, beta))
                alpha = max(alpha, v)
                if beta <= alpha:
                    break
            return v
        else:
            v = math.inf
            for n in node.find_children():
                v = min(v, self.minmax(n, depth - 1, True, alpha, beta))
                beta = min(beta, v)
                if beta <= alpha:
                    break
            return v

    def play(self, board):
        '''
        Choose the best move
        '''
        possible_moves = []
        for x in range(4):
            for y in range(4):
                for next_piece in range(16):
                    if board.check_if_move_valid(board.get_selected_piece(), x, y, next_piece):
                        print("trying move: ",
                              board.get_selected_piece(), x, y, next_piece)
                        possible_moves.append([board.get_selected_piece(), x, y, next_piece, self.minmax(create_node(board.make_move(
                            board.get_selected_piece(), x, y, next_piece, newboard=True)), self.depth, False, -math.inf, math.inf)])
        best_move = max(possible_moves, key=lambda x: x[4])[0:4]
        return board.make_move(best_move[0], best_move[1], best_move[2], best_move[3], newboard=True)

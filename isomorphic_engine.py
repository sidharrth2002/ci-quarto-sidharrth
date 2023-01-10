class IsomorphicEngine:
    def __init__(self):
        pass

    def check_if_board_is_symmetric(self, board):
        '''
        Check if board is symmetric
        '''
        for i in range(4):
            for j in range(4):
                if board[i][j] != board[3 - i][3 - j]:
                    return False
        return True

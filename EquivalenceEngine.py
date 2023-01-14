import numpy as np


class EquivalenceEngine:
    '''
    Get canonical representation of a board
    '''

    def __init__(self):
        # 2D arary of shape (16, 48)
        self.transpositions = [[0] * 48 for _ in range(16)]
        self.indexShuffle = [0, 5, 10, 15, 14,
                             13, 12, 9, 1, 6, 3, 2, 7, 11, 4, 8]

    def swapBit(self, input, x, y):
        if (x == y):
            return input
        xMask = 1 << x
        yMask = 1 << y
        xValue = (input & xMask) << (y - x)
        yValue = (input & yMask) >> (y - x)
        return input & ~xMask & ~yMask | xValue | yValue

    def getPiece(self, board, position):
        # unsigned right bit shift operator in python
        return (board >> (position << 2)) & 0xf

    def all_permutations(self, input, size, idx, permutations):
        for n in range(idx, size):
            if (idx == 3):
                permutations.append(input)
            self.all_permutations(self.swapBit(
                input, idx, n), size, idx+1, permutations)
        return permutations

    def get_transpositions(self):
        for piece in range(16):
            self.transpositions[piece][0] = piece
            permutations = self.all_permutations(piece, 4, 0, [])
            for n in range(24):
                self.transpositions[piece][n] = permutations[n]
            permutations = self.all_permutations(~piece & 0xf, 4, 0, [])
            for n in range(24, 48):
                self.transpositions[piece][n] = permutations[n-24]
        return self.transpositions

    def getMinTranspositionValues(self):
        minTranspositionValues = [0] * 16
        minTranspositions = []
        for n in range(16):
            min = 16
            elems = []
            transpositions = self.get_transpositions()
            for t in range(48):
                elem = transpositions[n][t]
                if (elem < min):
                    min = elem
                    elems = []
                    elems.append(t)
                elif (elem == min):
                    elems.append(t)
            minTranspositionValues[n] = min
            minTranspositions.append(elems)

        return minTranspositionValues, minTranspositions

    def convertBoard(self, board):
        '''
        Array to number
        '''
        newBoard = 0
        for n in range(16):
            newBoard |= n << (n * 4)
        return newBoard

    def getCanonicalRepresentation(self, board, turn):
        board = self.convertBoard(board)
        firstPiece = self.getPiece(board, self.indexShuffle[0])
        minTranspositionValues, minTranspositions = self.getMinTranspositionValues()
        signature = minTranspositionValues[firstPiece]
        transpositions = self.get_transpositions()
        ts = minTranspositions[firstPiece]
        print(ts)
        for n in range(turn):
            min = 16
            ts2 = []
            for t in ts:
                piece = self.getPiece(board, self.indexShuffle[n])
                posId = transpositions[piece][t]
                if posId == min:
                    ts2.append(t)
                elif posId < min:
                    min = posId
                    ts2 = []
                    ts2.append(t)
            ts = ts2
            signature = signature << 4 | min
        return signature


board = [[1, 1, 7, 1], [4, 0, 0, 2], [8, 9, 10, 11], [12, 13, 14, 15]]
board2 = [[1, 1, 2, 1], [4, 2, 3, 7], [8, 100, 10, 11], [12, 13, 14, 15]]

engine = EquivalenceEngine()
print(engine.getCanonicalRepresentation(board, 3))
print(engine.getCanonicalRepresentation(board2, 3))


positional_symmetries = [board, np.rot90(board, k=1, axes=(0, 1)), np.rot90(
    board, k=2, axes=(0, 1)), np.rot90(board, k=3, axes=(0, 1)), np.flip(board, axis=0), np.flip(np.rot90(board, k=1, axes=(0, 1)), axis=0), np.flip(np.rot90(board, k=2, axes=(0, 1)), axis=0), np.flip(np.rot90(board, k=3, axes=(0, 1)), axis=0), np.flip(board, axis=1), np.flip(np.rot90(board, k=1, axes=(0, 1)), axis=1), np.flip(np.rot90(board, k=2, axes=(0, 1)), axis=1), np.flip(np.rot90(board, k=3, axes=(0, 1)), axis=1), np.flip(board, axis=(0, 1)), np.flip(np.rot90(board, k=1, axes=(0, 1)), axis=(0, 1)), np.flip(np.rot90(board, k=2, axes=(0, 1)), axis=(0, 1)), np.flip(np.rot90(board, k=3, axes=(0, 1)), axis=(0, 1)), np.rot90(board, k=1, axes=(0, 1)), np.rot90(board, k=2, axes=(0, 1)), np.rot90(board, k=3, axes=(0, 1)), np.rot90(board, k=1, axes=(0, 1)), np.rot90(board, k=2, axes=(0, 1)), np.rot90(board, k=3, axes=(0, 1)), np.rot90(board, k=1, axes=(0, 1)), np.rot90(board, k=2, axes=(0, 1)), np.rot90(board, k=3, axes=(0, 1))]
print(len(positional_symmetries))

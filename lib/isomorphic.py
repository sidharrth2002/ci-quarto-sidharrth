import copy
import numpy as np


class BoardTransforms:
    """
    This class is used to transform a quarto board to normal form
    and to conduct comparisons between boards.
    """
    @staticmethod
    def compare_boards(board1, board2):
        """
        Compare two boards and return the one that is in normal form.
        """
        board1 = copy.deepcopy(board1)
        possible_transforms = [
            np.rot90(board1, 1),
            np.rot90(board1, 2),
            np.rot90(board1, 3),
            np.fliplr(board1),
            np.flipud(board1),
            # np.fliplr(np.rot90(board1, 1)),
            # np.fliplr(np.rot90(board1, 2)),
            # np.fliplr(np.rot90(board1, 3)),
            # np.flipud(np.rot90(board1, 1)),
            # np.flipud(np.rot90(board1, 2)),
            # np.flipud(np.rot90(board1, 3)),
            # np.fliplr(np.flipud(board1)),
            # np.flipud(np.fliplr(board1)),
            # np.fliplr(np.flipud(np.rot90(board1, 1))),
            # np.fliplr(np.flipud(np.rot90(board1, 2))),
            # np.fliplr(np.flipud(np.rot90(board1, 3))),
        ]

        if any(np.array_equal(board2, possible_transform) for possible_transform in possible_transforms):
            return True
        else:
            return False

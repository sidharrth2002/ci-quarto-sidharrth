def normal_form(self, board):
    '''
        Return normal form of board
        Two boards are equivalent if they reduce to the same normal form.
        The normal form is defined by the sequence of steps that generate it, and the sequence
        is chosen to ensure that all equivalent instances are reduced to the same normal form.
        '''
    normal_form = []

     # all of the positional symmetries are applied to the original instance, generating 32 equivalent, possibly distinct instances
     # each of these instances have and associated tag
     # this tag, or positional bit-mask, is a 17-bit string in which the ith bit is set if the ith square of the board is occuped
     positional_symmetries = [board, np.rot90(board, k=1, axes=(0, 1)), np.rot90(board, k=2, axes=(0, 1)), np.rot90(board, k=3, axes=(0, 1)), np.flip(board, axis=0), np.flip(
          np.rot90(board, k=1, axes=(0, 1)), axis=0), np.flip(np.rot90(board, k=2, axes=(0, 1)), axis=0), np.flip(np.rot90(board, k=3, axes=(0, 1)), axis=0), np.flip(board, axis=1), np.flip(
          np.rot90(board, k=1, axes=(0, 1)), axis=1), np.flip(np.rot90(board, k=2, axes=(0, 1)), axis=1), np.flip(np.rot90(board, k=3, axes=(0, 1)), axis=1), np.flip(board, axis=(0, 1)), np.flip(
          np.rot90(board, k=1, axes=(0, 1)), axis=(0, 1)), np.flip(np.rot90(board, k=2, axes=(0, 1)), axis=(0, 1)), np.flip(np.rot90(board, k=3, axes=(0, 1)), axis=(0, 1))]

      # those instances that share the largest bit-mask are candidates for normal form
      # each candidate instance is mapped with and XOR piece transformation.
      # the constant used is the value of the piece on the first occupied square
      # all 24 bitwise-permutaton piece transformations are applied to each of the candidate instances
      # the resulting instances are compared to each other with a string comparison
      # instance that results in the lexicographically least string is selected as normal form
      max_mask = 0
       for sym in positional_symmetries:
            mask = 0
            for i in range(4):
                for j in range(4):
                    mask = mask << 1
                    mask = mask | sym[i][j]
            if mask > max_mask:
                max_mask = mask

        for sym in positional_symmetries:
            if max_mask == 0:
                break
            mask = 0
            for i in range(4):
                for j in range(4):
                    mask = mask << 1
                    mask = mask | sym[i][j]
            if mask == max_mask:
                constant = sym[0][0]
                for piece_transformation in range(24):
                    temp = np.copy(sym)
                    for i in range(4):
                        for j in range(4):
                            temp[i][j] = self.piece_transformations[piece_transformation][sym[i][j]]
                    temp = temp.flatten()
                    temp = temp.tolist()
                    normal_form.append(temp)

        normal_form.sort()
        return normal_form[0]

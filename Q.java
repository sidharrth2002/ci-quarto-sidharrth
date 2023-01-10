public class Q {

    public static void main(String[] args) {
        System.out.println(countDraws(getStartBoard(), 0));
    }

    /**
     * Order of squares being filled, chosen to maximize the chance of an early win
     */
    private static int[] indexShuffle = { 0, 5, 10, 15, 14, 13, 12, 9, 1, 6, 3, 2, 7, 11, 4, 8 };

    /** Highest depth for using the lookup */
    private static final int MAX_LOOKUP_INDEX = 10;

    public static long countDraws(long board, int turn) {
        long signature = 0;
        if (turn < MAX_LOOKUP_INDEX) {
            signature = getSignature(board, turn);
            if (cache.get(turn).containsKey(signature))
                return cache.get(turn).get(signature);
        }
        int indexShuffled = indexShuffle[turn];
        long count = 0;
        for (int n = turn; n < 16; n++) {
            long newBoard = swap(board, indexShuffled, indexShuffle[n]);
            if (partialEvaluate(newBoard, indexShuffled))
                continue;
            if (turn == 15)
                count++;
            else
                count += countDraws(newBoard, turn + 1);
        }
        if (turn < MAX_LOOKUP_INDEX)
            cache.get(turn).put(signature, count);
        return count;
    }

    /** Get the canonical representation for this board and turn */
    private static long getSignature(long board, int turn) {
        int firstPiece = getPiece(board, indexShuffle[0]);
        long signature = minTranspositionValues[firstPiece];
        List<Integer> ts = minTranspositions.get(firstPiece);
        for (int n = 1; n < turn; n++) {
            int min = 16;
            List<Integer> ts2 = new ArrayList<>();
            for (int t : ts) {
                int piece = getPiece(board, indexShuffle[n]);
                int posId = transpositions[piece][t];
                if (posId == min) {
                    ts2.add(t);
                } else if (posId < min) {
                    min = posId;
                    ts2.clear();
                    ts2.add(t);
                }
            }
            ts = ts2;
            signature = signature << 4 | min;
        }
        return signature;
    }

    private static int getPiece(long board, int position) {
        return (int) (board >>> (position << 2)) & 0xf;
    }

    /** Only evaluate the relevant winning possibilities for a certain turn */
    private static boolean partialEvaluate(long board, int turn) {
        switch (turn) {
            case 15:
                return evaluate(board, masks[8]);
            case 12:
                return evaluate(board, masks[3]);
            case 1:
                return evaluate(board, masks[5]);
            case 3:
                return evaluate(board, masks[9]);
            case 2:
                return evaluate(board, masks[0]) || evaluate(board, masks[6]);
            case 11:
                return evaluate(board, masks[7]);
            case 4:
                return evaluate(board, masks[1]);
            case 8:
                return evaluate(board, masks[4]) || evaluate(board, masks[2]);
        }
        return false;
    }

    private static List<Map<Long, Long>> cache = new ArrayList<>();
    static {
        for (int i = 0; i < 16; i++)
            cache.add(new HashMap<>());
    }

    private static boolean evaluate(long board, long[] masks) {
        return _evaluate(board, masks) || _evaluate(~board, masks);
    }

    private static boolean _evaluate(long board, long[] masks) {
        for (long mask : masks)
            if ((board & mask) == mask)
                return true;
        return false;
    }

    private static long swap(long board, int x, int y) {
        if (x == y)
            return board;
        if (x > y)
            return swap(board, y, x);
        long xValue = (board & swapMasks[1][x]) << ((y - x) * 4);
        long yValue = (board & swapMasks[1][y]) >>> ((y - x) * 4);
        return board & swapMasks[0][x] & swapMasks[0][y] | xValue | yValue;
    }

    private static long getStartBoard() {
        long board = 0;
        for (long n = 0; n < 16; n++)
            board |= n << (n * 4);
        return board;
    }

    private static List<Integer> allPermutations(int input, int size, int idx, List<Integer> permutations) {
        for (int n = idx; n < size; n++) {
            if (idx == 3)
                permutations.add(input);
            allPermutations(swapBit(input, idx, n), size, idx + 1, permutations);
        }
        return permutations;
    }

    private static int swapBit(int in, int x, int y) {
        if (x == y)
            return in;
        int xMask = 1 << x;
        int yMask = 1 << y;
        int xValue = (in & xMask) << (y - x);
        int yValue = (in & yMask) >>> (y - x);
        return in & ~xMask & ~yMask | xValue | yValue;
    }

    private static int[][] transpositions = new int[16][48];
    static {
        for (int piece = 0; piece < 16; piece++) {
            transpositions[piece][0] = piece;
            List<Integer> permutations = allPermutations(piece, 4, 0, new ArrayList<>());
            for (int n = 1; n < 24; n++)
                transpositions[piece][n] = permutations.get(n);
            permutations = allPermutations(~piece & 0xf, 4, 0, new ArrayList<>());
            for (int n = 24; n < 48; n++)
                transpositions[piece][n] = permutations.get(n - 24);
        }
    }

    private static int[] minTranspositionValues = new int[16];
    private static List<List<Integer>> minTranspositions = new ArrayList<>();
    static {
        for (int n = 0; n < 16; n++) {
            int min = 16;
            List<Integer> elems = new ArrayList<>();
            for (int t = 0; t < 48; t++) {
                int elem = transpositions[n][t];
                if (elem < min) {
                    min = elem;
                    elems.clear();
                    elems.add(t);
                } else if (elem == min)
                    elems.add(t);
            }
            minTranspositionValues[n] = min;
            minTranspositions.add(elems);
        }
    }

    private static final long ROW_MASK = 1L | 1L << 4 | 1L << 8 | 1L << 12;
    private static final long COL_MASK = 1L | 1L << 16 | 1L << 32 | 1L << 48;
    private static final long FIRST_DIAG_MASK = 1L | 1L << 20 | 1L << 40 | 1L << 60;
    private static final long SECOND_DIAG_MASK = 1L << 12 | 1L << 24 | 1L << 36 | 1L << 48;

    private static long[][] masks = new long[10][4];
    static {
        for (int m = 0; m < 4; m++) {
            long row = ROW_MASK << (16 * m);
            for (int n = 0; n < 4; n++)
                masks[m][n] = row << n;
        }
        for (int m = 0; m < 4; m++) {
            long row = COL_MASK << (4 * m);
            for (int n = 0; n < 4; n++)
                masks[m + 4][n] = row << n;
        }
        for (int n = 0; n < 4; n++)
            masks[8][n] = FIRST_DIAG_MASK << n;
        for (int n = 0; n < 4; n++)
            masks[9][n] = SECOND_DIAG_MASK << n;
    }

    private static long[][] swapMasks;
    static {
        swapMasks = new long[2][16];
        for (int n = 0; n < 16; n++)
            swapMasks[1][n] = 0xfL << (n * 4);
        for (int n = 0; n < 16; n++)
            swapMasks[0][n] = ~swapMasks[1][n];
    }
}
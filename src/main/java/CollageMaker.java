import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

public class CollageMaker {

    private static final int UNPLACED_SCORE = Integer.MAX_VALUE / 500;
    private static final int MAX_LARGE_IMAGE_SIZE = 300;
    private static final long MAX_MILLIS = 8500;

    private Random m_rand;

    private static int intersect(int a, int b, int c, int d) {
        int from = Math.max(a, c);
        int to = Math.min(b, d);
        return from < to ? to - from : 0;
    }

    private static int[][] topCoderRescale(int[][] pixels, int newW, int newH) {
        List<Integer> origRList = new ArrayList<Integer>();
        List<Integer> newRList = new ArrayList<Integer>();
        List<Integer> intrRList = new ArrayList<Integer>();
        List<Integer> origCList = new ArrayList<Integer>();
        List<Integer> newCList = new ArrayList<Integer>();
        List<Integer> intrCList = new ArrayList<Integer>();

        int W = pixels[0].length;
        int H = pixels.length;

        for (int origR = 0; origR < H; origR++) {
            int r1 = origR * newH, r2 = r1 + newH;
            for (int newR = 0; newR < newH; newR++) {
                int r3 = newR * H, r4 = r3 + H;
                int intr = intersect(r1, r2, r3, r4);
                if (intr > 0) {
                    origRList.add(origR);
                    newRList.add(newR);
                    intrRList.add(intr);
                }
            }
        }

        for (int origC = 0; origC < W; origC++) {
            int c1 = origC * newW, c2 = c1 + newW;
            for (int newC = 0; newC < newW; newC++) {
                int c3 = newC * W, c4 = c3 + W;
                int intr = intersect(c1, c2, c3, c4);
                if (intr > 0) {
                    origCList.add(origC);
                    newCList.add(newC);
                    intrCList.add(intr);
                }
            }
        }

        int[][] res = new int[newH][newW];
        for (int i = 0; i < origRList.size(); i++) {
            int origR = origRList.get(i);
            int newR = newRList.get(i);
            int intrR = intrRList.get(i);

            for (int j = 0; j < origCList.size(); j++) {
                int origC = origCList.get(j);
                int newC = newCList.get(j);
                int intrC = intrCList.get(j);

                res[newR][newC] += intrR * intrC * pixels[origR][origC];
            }
        }

        for (int r = 0; r < newH; r++) {
            for (int c = 0; c < newW; c++) {
                res[r][c] = (2 * res[r][c] + H * W) / (2 * H * W);
            }
        }

        return res;
    }

    private static class TimerAndLogger {

        private final long m_startTime;

        private TimerAndLogger() {
            m_startTime = System.currentTimeMillis();
        }

        private void log(String fmt, Object... args) {
            System.err.println(deltaTime() + " " + String.format(fmt, args));
            System.err.flush();
        }

        private long deltaTime() {
            return System.currentTimeMillis() - m_startTime;
        }

        private boolean isEndTime() {
            return deltaTime() >= MAX_MILLIS;
        }
    }

    private static class Board {

        private final MyImage m_sourceImage;
        private final Tile[] m_tiles;
        private final TimerAndLogger m_logger;
        private int m_score;
        private final Random m_rand = new Random(6);

        private Board(TimerAndLogger logger, MyImage sourceImage, Tile[] tiles) {
            m_logger = logger;
            m_sourceImage = sourceImage;
            m_tiles = tiles;
        }

        public int[] toArray() {
            int[] ret = new int[m_tiles.length * 4];
            int i = 0;
            for (Tile tile : m_tiles) {
                for (int val : tile.box()) {
                    ret[i++] = val;
                }
            }
            return ret;
        }

        private Iterator<Coord> getRandomFreePlacements(final int deltaWidth, final int deltaHeight) {

            return new Iterator<CollageMaker.Coord>() {

                final Iterator<Coord> tileList;
                private Coord m_onDeck;
                {
                    List<Coord> tl = getAllCoords(height(), width(), 1, 1);
                    Collections.shuffle(tl, m_rand);
                    tileList = tl.iterator();
                    advance();
                }

                private void advance() {
                    while (tileList.hasNext()) {
                        m_onDeck = tileList.next();
                        if (canPlace(m_onDeck.m_row, m_onDeck.m_col, deltaWidth, deltaHeight)) {
                            break;
                        }
                    }
                }

                @Override
                public boolean hasNext() {
                    return m_onDeck != null;
                }

                @Override
                public Coord next() {
                    Coord ret = m_onDeck;
                    m_onDeck = null;
                    advance();
                    return ret;
                }

                @Override
                public void remove() {

                }
            };
        }

        private boolean swap(int tile1index, int tile2index) {
            Tile tile1 = m_tiles[tile1index];
            Tile tile2 = m_tiles[tile2index];
            if (!tile1.isPlaced() && !tile2.isPlaced()) {
                return false;
            }

            int prevScore = m_score;
            int[] box1 = tile1.box();
            int[] box2 = tile2.box();
            placeFromBox(tile1index, box2);
            placeFromBox(tile2index, box1);
            int postScore = m_score;

            if (prevScore <= postScore) {
                // revert
                placeFromBox(tile1index, box1);
                placeFromBox(tile2index, box2);
                if (m_score != prevScore) {
                    throw new IllegalStateException(String.format("Prevscore %d != new score %d after replacement", prevScore, m_score));
                }
                return false;
            } else {
                return true;
            }
        }

        private int height() {
            return m_sourceImage.m_height;
        }

        private int width() {
            return m_sourceImage.m_width;
        }

        public int[] place(int tilePos, int row, int col, int width, int height) {
            return place(tilePos, row, col, width, height, false);
        }

        private int tileConvering(int row, int col, boolean throwExceptionIfMoreThanOne) {
            int ret = -1;
            for (Tile t : m_tiles) {
                if (t.containsPixel(row, col, 1, 1)) {
                    if (!throwExceptionIfMoreThanOne) {
                        return t.m_tileNumber;
                    }
                    if (ret != -1)
                        throw new IllegalStateException(String.format("Tiles %s %s and %s %s already cover (%d,%d)", m_tiles[ret],
                                Arrays.toString(m_tiles[ret].box()), m_tiles[t.m_tileNumber], Arrays.toString(m_tiles[t.m_tileNumber].box()), row,
                                col));
                    ret = t.m_tileNumber;
                }
            }
            return ret;
        }

        private int tileConvering(int row, int col) {
            return tileConvering(row, col, false);
        }

        public int[] place(int tilePos, int row, int col, int width, int height, boolean doLog) {
            width = Math.min(width, width() - col);
            height = Math.min(height, height() - row);
            Tile tile = m_tiles[tilePos];
            int prevScore = tile.score(m_sourceImage);
            if (tile.isPlaced())
                m_score -= tile.score(m_sourceImage);
            tile.change(row, col, width, height);
            if (tile.isPlaced())
                m_score += tile.score(m_sourceImage);
            if (doLog)
                m_logger.log("Placed tile %s score changed by %d", m_tiles[tilePos], tile.score(m_sourceImage) - prevScore);
            return new int[] { width, height };
        }

        public List<Tile> createTileList() {
            return new ArrayList<CollageMaker.Tile>(Arrays.asList(m_tiles));
        }

        private void resetAllTiles() {
            for (int tilePos = 0; tilePos < m_tiles.length; tilePos++) {
                place(tilePos, -1, -1, -1, -1);
            }
            if (m_score != 0) {
                throw new IllegalStateException("When resetting board score should be 0, not " + m_score);
            }
        }

        public int[] placeFromBox(int tileNumber, int[] box) {
            return place(tileNumber, box[0], box[1], box[3] - box[1] + 1, box[2] - box[0] + 1, false);
        }

        private boolean canPlace(int row, int col, int width, int height) {
            // optimize l8r
            if (row + height >= height()) {
                return false;
            }
            if (col + width >= width()) {
                return false;
            }
            for (Tile t : m_tiles) {
                if (t.isPlaced() && t.containsPixel(row, col, width, height)) {
                    return false;
                }
            }
            return true;
        }

        public List<Tile> getFreeTiles() {
            List<Tile> ret = new ArrayList<CollageMaker.Tile>();
            for (Tile t : m_tiles) {
                if (!t.isPlaced()) {
                    ret.add(t);
                }
            }
            Collections.shuffle(ret, m_rand);
            return ret;
        }
    }

    private static class Tile {
        private final MyImage m_image;
        private int m_row, m_col;
        private final Map<String, Integer> m_cachedScore = new HashMap<String, Integer>();
        private final int m_tileNumber;
        private MyImage m_scaledImage;

        private Tile(int tileNumber, MyImage image) {
            m_tileNumber = tileNumber;
            this.m_image = image;
            m_row = -1;
            m_col = -1;
        }

        private boolean containsPixel(int row, int col, int width, int height) {
            int[] b1 = box();
            int[] b2 = new int[] { row, col, row + height - 1, col + width - 1 };
            if (b1[0] > b2[2]) {
                return false;
            }
            if (b1[2] < b2[0]) {
                return false;
            }
            if (b1[1] > b2[3]) {
                return false;
            }
            if (b1[3] < b2[1]) {
                return false;
            }
            return true;
        }

        private void change(int row, int col, int width, int height) {
            if (row < 0 || col < 0 || width < 1 || height < 1) {
                m_scaledImage = null;
                m_row = -1;
                m_col = -1;
            } else {
                m_row = row;
                m_col = col;
                m_scaledImage = m_image.resize(width, height);
            }
        }

        @Override
        public String toString() {
            String dim = (m_scaledImage == null) ? "n/a" : String.format("(%d,%d)", m_scaledImage.m_width, m_scaledImage.m_height);
            return String.format("<Tile: #%03d, Img: %s, Pos: (%d,%d), Dim: %s>", m_tileNumber, m_image, m_row, m_col, dim);
        }

        public boolean isPlaced() {
            return m_row >= 0 && m_col >= 0 && m_scaledImage != null;
        }

        @Override
        public int hashCode() {
            final int prime = 31;
            int result = 1;
            result = prime * result + m_tileNumber;
            return result;
        }

        @Override
        public boolean equals(Object obj) {
            if (this == obj)
                return true;
            if (obj == null)
                return false;
            if (getClass() != obj.getClass())
                return false;
            Tile other = (Tile) obj;
            if (m_tileNumber != other.m_tileNumber)
                return false;
            return true;
        }

        private int score(MyImage sourceImage, final int row, final int col, int width, int height) {
            String key = row + "x" + col + "_" + width + "x" + height;
            if (m_cachedScore.containsKey(key)) {
                return m_cachedScore.get(key);
            }
            MyImage meScaled = m_image.resize(width, height);
            double ret = 0;
            for (int myRow = 0; myRow < meScaled.m_height; myRow++) {
                if (myRow + row >= sourceImage.m_height) {
                    break;
                }
                for (int myCol = 0; myCol < meScaled.m_width; myCol++) {
                    if (myCol + col >= sourceImage.m_width) {
                        break;
                    }
                    ret += Math.pow(sourceImage.m_imageData[myRow + row][myCol + col] - meScaled.m_imageData[myRow][myCol], 2);
                }
            }
            int ret2 = (int) Math.sqrt(ret);
            m_cachedScore.put(key, ret2);
            return ret2;
        }

        private int score(MyImage sourceImage) {
            if (!isPlaced()) {
                return UNPLACED_SCORE;
            }
            return score(sourceImage, m_row, m_col, m_scaledImage.m_width, m_scaledImage.m_height);
        }

        public int[] box() {
            if (isPlaced()) {
                return new int[] { m_row, m_col, m_row + height() - 1, m_col + width() - 1 };
            } else {
                return new int[] { -1, -1, -1, -1 };
            }
        }

        public int width() {
            return m_scaledImage.m_width;
        }

        public int height() {
            return m_scaledImage.m_height;
        }

    }

    private static class MyImage {
        private final int[][] m_imageData;
        private final Map<String, MyImage> cached = new HashMap<String, CollageMaker.MyImage>();
        private final int m_width;
        private final int m_height;

        private MyImage(int[][] imageData) {
            m_imageData = imageData;
            m_height = m_imageData.length;
            m_width = m_imageData[0].length;
        }

        private MyImage resize(int width, int height) {
            String key = width + "x" + height;
            if (cached.containsKey(key)) {
                return cached.get(key);
            }
            MyImage ret = new MyImage(topCoderRescale(m_imageData, width, height));
            cached.put(key, ret);
            return ret;
        }

        @Override
        public String toString() {
            return String.format("<Img: (%d,%d)>", m_width, m_height);
        }

    }

    private static List<MyImage> parseData(int[] data) {
        List<MyImage> ret = new ArrayList<CollageMaker.MyImage>();
        int i = 0;
        for (int imageNumber = 0; imageNumber < 201; imageNumber++) {
            int height = data[i++];
            int width = data[i++];
            int[][] imageData = new int[height][width];
            for (int row = 0; row < height; row++) {
                int[] rowData = new int[width];
                for (int col = 0; col < width; col++) {
                    rowData[col] = data[i++];
                }
                imageData[row] = rowData;
            }
            ret.add(new MyImage(imageData));
        }
        return ret;
    }

    public int[] compose(int[] data) {
        m_rand = new Random();
        m_rand.setSeed(1);
        TimerAndLogger timer = new TimerAndLogger();
        timer.log("Startup");
        List<MyImage> allImages = parseData(data);
        MyImage sourceImage = allImages.remove(0);
        timer.log("Source image: %s", sourceImage);
        Tile[] tiles = new Tile[allImages.size()];
        for (int i = 0; i < tiles.length; i++) {
            tiles[i] = new Tile(i, allImages.get(i));
        }
        timer.log("Number tiles: %d", tiles.length);
        Board board = new Board(timer, sourceImage, tiles);
        timer.log("Created board");
        doPlacement(board);
        int[] ret = board.toArray();
        timer.log("Returning array");
        return ret;
    }

    private void doInitialPlacement(Board board, int deltaWidth, int deltaHeight) {
        List<Tile> tiles = board.createTileList();
        Collections.shuffle(tiles, m_rand);
        // initial placement
        int row = 0;
        while (row < board.height()) {
            int col = 0;
            int deltaH = 0;
            while (col < board.width()) {
                int bestScore = Integer.MAX_VALUE;
                Tile bestTile = null;
                for (Tile t : tiles) {
                    int score = t.score(board.m_sourceImage, row, col, deltaWidth, deltaHeight);
                    if (score < bestScore) {
                        bestScore = score;
                        bestTile = t;
                    }
                }
                if (bestTile == null) {
                    throw new IllegalStateException(String.format("For position (%d,%d) with dim (%d,%d) cannot find best tile out of %d", row, col,
                            deltaWidth, deltaHeight, tiles.size()));
                }
                tiles.remove(bestTile);
                int width = Math.min(deltaWidth, board.width() - col);
                int height = Math.min(deltaHeight, board.height() - row);
                int[] placedWidthHeight = board.place(bestTile.m_tileNumber, row, col, width, height, false);
                // board.m_logger.log("Placed tile %s with score %d", bestTile, bestScore);
                col += placedWidthHeight[0];
                deltaH = placedWidthHeight[1];
            }
            row += deltaH;
        }
        board.m_logger.log("Done with initial placement, score: %d", board.m_score);
    }

    private void doInitialPlacement(Board board) {

        int deltaWidth = board.m_sourceImage.m_width / 13 + 1;
        int deltaHeight = board.m_sourceImage.m_height / 13 + 1;

        int bestDeltaWidth = -1;
        int bestDeltaHeight = -1;
        int bestScore = Integer.MAX_VALUE;

        for (int dw = Math.max(deltaWidth - 2, 10); dw < Math.min(deltaWidth + 3, 30); dw += 2) {
            for (int dh = Math.max(deltaHeight - 2, 10); dh < Math.min(deltaHeight + 3, 30); dh += 2) {
                if (board.m_logger.deltaTime() > MAX_MILLIS / 2) {
                    break;
                }
                int totalAcross = board.m_sourceImage.m_width / dw;
                if (totalAcross * deltaWidth != board.m_sourceImage.m_width)
                    totalAcross++;
                int totalUpDown = board.m_sourceImage.m_height / dh;
                if (totalUpDown * deltaWidth != board.m_sourceImage.m_height)
                    totalUpDown++;
                if (totalAcross * totalUpDown > 190) {
                    board.m_logger.log("Too many pieces with dim (%d,%d) : %d*%d=%d", dw, dh, totalAcross, totalUpDown, totalAcross * totalUpDown);
                    continue;
                }
                board.resetAllTiles();
                try {
                    doInitialPlacement(board, dw, dh);
                } catch (IllegalStateException ex) {
                    board.m_logger.log("Skipping size as %s", ex);
                    continue;
                }
                if (board.m_score < bestScore) {
                    bestScore = board.m_score;
                    bestDeltaWidth = dw;
                    bestDeltaHeight = dh;
                    board.m_logger.log("Best board score dim (%d,%d) : %d*%d=%d is %d", bestDeltaWidth, bestDeltaHeight, totalAcross, totalUpDown,
                            totalAcross * totalUpDown, bestScore);
                }
            }
        }
        board.resetAllTiles();
        deltaWidth = bestDeltaWidth;
        deltaHeight = bestDeltaHeight;
        doInitialPlacement(board, deltaWidth, deltaHeight);

        int totalAcross = board.m_sourceImage.m_width / deltaWidth;
        if (totalAcross * deltaWidth != board.m_sourceImage.m_width)
            totalAcross++;
        int totalUpDown = board.m_sourceImage.m_height / deltaHeight;
        if (totalUpDown * deltaWidth != board.m_sourceImage.m_height)
            totalUpDown++;

        board.m_logger.log("For source image %s using (%d,%d) sized tiles for a total of %d*%d=%d tiles", board.m_sourceImage, deltaWidth,
                deltaHeight, totalAcross, totalUpDown, totalAcross * totalUpDown);

    }

    private void doPlacement(Board board) {
        placeRandomly(board);
        board.m_logger.log("Done placing random board");
        // doInitialPlacement(board);
        doOptimization1(board);
    }

    private Set<Coord> makeAvailableUppers(Board board, int deltaWidth, int deltaHeight) {
        Set<Coord> availableUpperCorners = new HashSet<Coord>();
        for (int row = 0; row < board.m_sourceImage.m_height - deltaHeight; row += deltaHeight) {
            for (int col = 0; col < board.m_sourceImage.m_width - deltaWidth; col += deltaWidth) {
                Coord c = new Coord(row, col);
                // board.m_logger.log("Available: %s", c);
                availableUpperCorners.add(c);
            }
        }
        return availableUpperCorners;
    }

    private static class Coord {

        private final int m_row, m_col;

        public Coord(int row, int col) {
            m_row = row;
            m_col = col;
        }

        @Override
        public int hashCode() {
            final int prime = 31;
            int result = 1;
            result = prime * result + m_col;
            result = prime * result + m_row;
            return result;
        }

        @Override
        public boolean equals(Object obj) {
            if (this == obj)
                return true;
            if (obj == null)
                return false;
            if (getClass() != obj.getClass())
                return false;
            Coord other = (Coord) obj;
            if (m_col != other.m_col)
                return false;
            if (m_row != other.m_row)
                return false;
            return true;
        }

        @Override
        public String toString() {
            return String.format("(%d,%d)", m_row, m_col);
        }

    }

    public static List<Coord> getAllCoords(int endRow, int endCol, int deltaWidth, int deltaHeight) {
        return getAllCoords(new Coord(0, 0), new Coord(endRow, endCol), deltaWidth, deltaHeight);
    }

    public static List<Coord> getAllCoords(Coord start, Coord end, int deltaWidth, int deltaHeight) {
        List<Coord> ret = new ArrayList<CollageMaker.Coord>();
        for (int row = start.m_row; row <= end.m_row; row += deltaHeight) {
            for (int col = start.m_col; col < end.m_col; col += deltaWidth) {
                ret.add(new Coord(row, col));
            }
        }
        return ret;
    }

    private class RandomPlacer {
        private final Board m_board;

        private RandomPlacer(Board board) {
            m_board = board;
        }

        private int doPickFulls(Iterator<Coord> it, int width, int height, int numberSlots) {
            m_board.m_logger.log("Picking %d places for size (%d, %d)", numberSlots, width, height);
            List<Tile> freeTiles = new ArrayList<CollageMaker.Tile>();
            List<Tile> allTiles = m_board.getFreeTiles();
            while (!allTiles.isEmpty() && freeTiles.size() != numberSlots) {
                freeTiles.add(allTiles.remove(0));
            }
            if (freeTiles.isEmpty()) {
                return 0;
            }
            List<Coord> slots = new ArrayList<CollageMaker.Coord>();

            while (it.hasNext() && slots.size() != freeTiles.size()) {
                Coord c = it.next();
                if (m_board.canPlace(c.m_row, c.m_col, width, height)) {
                    if (hasOverlap(slots, c, width, height)) {
                        m_board.m_logger.log("Cannot select slot %s with other slots %s", c, slots);
                        continue;
                    }
                    slots.add(c);
                }
            }
            if (slots.isEmpty() || freeTiles.isEmpty()) {
                m_board.m_logger.log("No slots or tiles available");
                return 0;
            }
            while (freeTiles.size() != slots.size()) {
                freeTiles.remove(0);
            }
            m_board.m_logger.log("Optimizing %d slots", slots.size());
            optimizePicks(slots, freeTiles, width, height);
            return slots.size();
        }

        public boolean hasOverlap(List<Coord> slots, Coord c, int width, int height) {
            int[] b1 = new int[] { c.m_row, c.m_col, c.m_row + height - 1, c.m_col + width - 1 };
            for (Coord s : slots) {
                int[] b2 = new int[] { s.m_row, s.m_col, s.m_row + height - 1, s.m_col + width - 1 };
                if (b1[0] > b2[2]) {
                    continue;
                }
                if (b1[2] < b2[0]) {
                    continue;
                }
                if (b1[1] > b2[3]) {
                    continue;
                }
                if (b1[3] < b2[1]) {
                    continue;
                }
                return true;
            }
            return false;
        }

        private void optimizePicks(List<Coord> slots, List<Tile> tiles, int width, int height) {
            // first put them all in place
            m_board.m_logger.log("Placing these tiles: %s into slots: %s", tiles, slots);
            for (int i = 0; i < slots.size(); i++) {
                m_board.place(tiles.get(i).m_tileNumber, slots.get(i).m_row, slots.get(i).m_col, width, height);
            }
            for (Tile t : tiles) {
                if (!t.isPlaced()) {
                    throw new IllegalStateException("Did not place tile " + t);
                }
            }
            int preScore = m_board.m_score;
            int totalSwapped = 0;
            while (true) {
                int numberSwapped = 0;
                for (int i = 0; i < tiles.size() - 1; i++) {
                    for (int j = i + 1; j < tiles.size(); j++) {
                        if (m_board.swap(tiles.get(i).m_tileNumber, tiles.get(j).m_tileNumber)) {
                            numberSwapped++;
                            break;
                        }
                    }
                }
                if (numberSwapped == 0)
                    break;
                totalSwapped += numberSwapped;
            }
            int postScore = m_board.m_score;
            m_board.m_logger.log("Placed tiles: %s", tiles);
            m_board.m_logger.log("Score went from %d to %d after doing %d swaps", preScore, postScore, totalSwapped);
        }

    }

    private void placeRandomly(Board board) {
        board.m_logger.log("Starting random placement");

        int deltaWidth = 2 * ((board.m_sourceImage.m_width / 8 + 1) / 2);
        int deltaHeight = 2 * ((board.m_sourceImage.m_height / 8 + 1) / 2);
        RandomPlacer placer = new RandomPlacer(board);

        Iterator<Coord> it = board.getRandomFreePlacements(deltaWidth, deltaHeight);
        int totalPlaced = 0;
        while (true) {
            int thisPlaced = placer.doPickFulls(it, deltaWidth, deltaHeight, 5);
            if (thisPlaced == 0)
                break;
            totalPlaced += thisPlaced;
            board.m_logger.log("Total placed: %d", totalPlaced);
        }
        board.m_logger.log("End Total placed: %d", totalPlaced);

        int r = 198;
        int c = 212;
        int tileIndex = board.tileConvering(r, c, true);
        board.m_logger.log("At (%d,%d) have tile %s", r, c, board.m_tiles[tileIndex]);

        deltaWidth /= 2;
        deltaHeight /= 2;

        board.m_logger.log("New dimensions (%d,%d)", deltaWidth, deltaHeight);

        it = board.getRandomFreePlacements(deltaWidth, deltaHeight);
        while (true) {
            int thisPlaced = placer.doPickFulls(it, deltaWidth, deltaHeight, 5);
            if (thisPlaced == 0)
                break;
            totalPlaced += thisPlaced;
            board.m_logger.log("Total placed: %d", totalPlaced);
        }
        board.m_logger.log("End Total placed: %d", totalPlaced);

        deltaWidth /= 2;
        deltaHeight /= 2;

        board.m_logger.log("New dimensions (%d,%d)", deltaWidth, deltaHeight);

        it = board.getRandomFreePlacements(deltaWidth, deltaHeight);
        while (true) {
            int thisPlaced = placer.doPickFulls(it, deltaWidth, deltaHeight, 5);
            if (thisPlaced == 0)
                break;
            totalPlaced += thisPlaced;
            board.m_logger.log("Total placed: %d", totalPlaced);
        }
        board.m_logger.log("End Total placed: %d", totalPlaced);

        deltaWidth =1;
        deltaHeight = 1;

        board.m_logger.log("New dimensions (%d,%d)", deltaWidth, deltaHeight);

        it = board.getRandomFreePlacements(deltaWidth, deltaHeight);
        while (true) {
            int thisPlaced = placer.doPickFulls(it, deltaWidth, deltaHeight, 5);
            if (thisPlaced == 0)
                break;
            totalPlaced += thisPlaced;
            board.m_logger.log("Total placed: %d", totalPlaced);
        }
        board.m_logger.log("End Total placed: %d", totalPlaced);

    }

    private void fillInAllEmpties(Board board) {
        List<Coord> missing = new ArrayList<CollageMaker.Coord>();
        for (int row = 0; row < board.height(); row++) {
            for (int col = 0; col < board.width(); col++) {
                Coord c = new Coord(row, col);
                int tilePos = board.tileConvering(c.m_row, c.m_col);
                if (tilePos == -1) {
                    missing.add(c);
                }
            }
        }
        List<Tile> freeTiles = board.getFreeTiles();
        board.m_logger.log("Missing %d empties to fill with %d", missing.size(), freeTiles.size());
        board.m_logger.log("All missing: %s", missing);
        List<Coord> missing2 = new ArrayList<CollageMaker.Coord>();
        while (!missing.isEmpty() && !freeTiles.isEmpty()) {
            board.m_logger.log("Missing: %d, FT: %d", missing.size(), freeTiles.size());
            Coord c = missing.remove(0);
            Set<Coord> nextFill = new HashSet<CollageMaker.Coord>();
            nextFill.add(c);
            Set<Integer> rowVals = new HashSet<Integer>();
            rowVals.add(c.m_row);
            for (Coord c2 : missing) {
                if (c2.m_col == c.m_col) {
                    rowVals.add(c2.m_row);
                    nextFill.add(c2);
                }
            }
            if (rowVals.isEmpty()) {
                board.m_logger.log("WARNING: cannot place %s", c);
                missing2.add(c);
                continue;
            }
            if (rowVals.size() == Collections.max(rowVals) - Collections.min(rowVals) + 1) {
                // can fill this all in
                Tile ft = freeTiles.remove(0);
                board.m_logger.log("Growing1 %s with %s from %d to %d => %d", ft, nextFill, Collections.min(rowVals), Collections.max(rowVals),
                        rowVals.size());
                board.place(ft.m_tileNumber, Collections.min(rowVals), c.m_col, 1, rowVals.size(), true);
                missing.removeAll(nextFill);
            } else {
                board.m_logger.log("WARNING: bad size place %s : %d : %s", c, rowVals.size(), rowVals);
            }
        }
        board.tileConvering(192, 59, true);
        board.m_logger.log("Free tiles: %d, missing: %d", freeTiles.size(), missing2.size());
        while (!missing2.isEmpty() && !freeTiles.isEmpty()) {
            Coord c = missing2.remove(0);
            Set<Coord> nextFill = new HashSet<CollageMaker.Coord>();
            nextFill.add(c);
            Set<Integer> colVals = new HashSet<Integer>();
            colVals.add(c.m_col);
            for (Coord c2 : missing) {
                if (c2.m_row == c.m_row) {
                    colVals.add(c2.m_col);
                    nextFill.add(c2);
                }
            }
            if (colVals.isEmpty()) {
                board.m_logger.log("WARNING: cannot place %s", c);
                missing2.add(c);
                continue;
            }
            if (colVals.size() == Collections.max(colVals) - Collections.min(colVals)) {
                // can fill this all in
                Tile ft = freeTiles.remove(0);
                board.m_logger.log("Growing2 %s", ft);
                board.place(ft.m_tileNumber, c.m_row, Collections.min(colVals), colVals.size(), 1, true);
                missing.removeAll(nextFill);
            }
        }
    }

    private void growSides(Board board, int deltaWidth, int deltaHeight) {
        int row = 0;
        Set<Integer> alreadyStretched = new HashSet<Integer>();
        while (row < board.height() - 1) {
            int col = board.width() - 1;
            int tilePos = board.tileConvering(row, col, true);
            while (tilePos == -1 && col >= 0) {
                col -= 1;
                tilePos = board.tileConvering(row, col, true);
            }
            if (tilePos == -1)
                throw new IllegalStateException(String.format("Cannot find cover at (%d,%d)", row, col));
            if (alreadyStretched.contains(tilePos)) {
                row += 1;
                continue;
            }
            Tile t = board.m_tiles[tilePos];
            if (t.box()[3] == board.width() - 1) {
                board.m_logger.log("Skipping %s :%s as touches edge", t, Arrays.toString(t.box()));
                row = t.box()[2] + 1;
                continue;
            }
            String prev = t.toString();
            if (!board.canPlace(t.m_row, t.m_col, board.width() - t.m_col, t.height())) {
                board.m_logger.log("Warning: cannot horz stretch %s", t);
            } else {
                board.place(tilePos, t.m_row, t.m_col, board.width() - t.m_col, t.height());
                board.m_logger.log("Stretched horz %s to %s : %s (row=%d,col=%d)", prev, t, Arrays.toString(t.box()), row, col);
                alreadyStretched.add(t.m_tileNumber);
            }
            row += 1;
        }
        int col = 0;
        while (col < board.width() - 1) {
            row = board.height() - 1;
            int tilePos = board.tileConvering(row, col);
            while (tilePos == -1 && row >= 0) {
                row -= 1;
                tilePos = board.tileConvering(row, col);
            }
            if (tilePos == -1)
                throw new IllegalStateException(String.format("Cannot find cover at (%d,%d)", row, col));
            Tile t = board.m_tiles[tilePos];
            if (!board.canPlace(t.m_row, t.m_col, t.width(), board.height() - t.m_row)) {
                board.m_logger.log("Warning: cannot vert stretch %s", t);
            } else {
                board.place(tilePos, t.m_row, t.m_col, t.width(), board.height() - t.m_row);
                board.m_logger.log("Stretched vert %s : %s", t, Arrays.toString(t.box()));
            }
            col += t.width();
        }
    }

    private void removePlacement(Set<int[]> available, int[] placed) {
        if (!available.remove(placed)) {
            // throw new IllegalStateException("Did not find " + Arrays.toString(placed) + " in available");
        }
    }

    private void doOptimization2(Board board) {
        // bump each row over to the right edge and see if can better place
        // just do the first row
        int row = 0;
        while (row < board.m_sourceImage.m_height) {
            Set<Integer> allTilePosSet = new LinkedHashSet<Integer>();
            List<Integer> allTilePos = new ArrayList<Integer>();
            for (int col = 0; col < board.m_sourceImage.m_width; col++) {
                int tileNum = board.tileConvering(row, col);
                if (allTilePosSet.add(tileNum))
                    allTilePos.add(tileNum);
            }
            Tile startTile = board.m_tiles[allTilePos.get(0)];
            int typicalWidth = startTile.width();
            int height = startTile.height();
            Collections.reverse(allTilePos);
            Tile endTile = board.m_tiles[allTilePos.remove(0).intValue()];
            board.m_logger.log("End tile is %s : %s", endTile, Arrays.toString(endTile.box()));
            board.place(endTile.m_tileNumber, -1, -1, -1, -1);
            int col = board.m_sourceImage.m_width - typicalWidth;
            while (!allTilePos.isEmpty()) {
                Tile tile = board.m_tiles[allTilePos.remove(0)];
                int[] box = tile.box();
                box[1] = col;
                box[3] = box[1] + typicalWidth - 1;
                board.placeFromBox(tile.m_tileNumber, box);
                board.m_logger.log("Moved tile to %s : %s", tile, Arrays.toString(tile.box()));
                col -= typicalWidth;
            }
            board.place(endTile.m_tileNumber, row, 0, col + typicalWidth, height);
            board.m_logger.log("Moved tile to %s : %s", endTile, Arrays.toString(endTile.box()));
            row += height;
        }
    }

    private void doOptimization1(Board board) {
        while (!board.m_logger.isEndTime()) {
            int numberChanged = 0;
            for (int t1 = 0; t1 < 199; t1++) {
                for (int t2 = t1 + 1; t2 < 199; t2++) {
                    if (board.m_logger.isEndTime()) {
                        break;
                    }
                    if (board.swap(t1, t2)) {
                        // board.m_logger.log("Board score after swap: %d", board.m_score);
                        numberChanged++;
                        break;
                    } else {
                        // board.m_logger.log("Board score no    swap: %d", board.m_score);
                    }
                }
                if (board.m_logger.isEndTime()) {
                    break;
                }
            }
            board.m_logger.log("Moved %d pieces, score: %d", numberChanged, board.m_score);
            if (numberChanged == 0)
                break;
        }
    }
}


import java.util.ArrayList;
import java.util.List;

public class CollageMaker {

    private static class Board {

        private final MyImage m_sourceImage;
        private final Tile[] m_tiles;

        private Board(MyImage sourceImage, Tile[] tiles) {
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
    }

    private static class Tile {
        private final MyImage m_image;
        private int m_row, m_col, m_width, m_height;

        private Tile(MyImage image) {
            this.m_image = image;
            m_row = -1;
            m_col = -1;
            m_width = -1;
            m_height = -1;
        }

        public boolean isPlaced() {
            return m_row != -1 && m_col != -1 && m_width != -1 && m_height != -1;
        }

        public int[] box() {
            if (isPlaced()) {
                return new int[] { m_row, m_col, m_width, m_height };
            } else {
                return new int[] { -1, -1, -1, -1 };
            }
        }

        public int getRow() {
            return m_row;
        }

        public void setRow(int row) {
            m_row = row;
        }

        public int getCol() {
            return m_col;
        }

        public void setCol(int col) {
            m_col = col;
        }

        public int getWidth() {
            return m_width;
        }

        public void setWidth(int width) {
            m_width = width;
        }

        public int getHeight() {
            return m_height;
        }

        public void setHeight(int height) {
            m_height = height;
        }

        public MyImage getImage() {
            return m_image;
        }
    }

    private static class MyImage {
        private final int[][] m_imageData;

        private MyImage(int[][] imageData) {
            m_imageData = imageData;
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
        List<MyImage> allImages = parseData(data);
        MyImage sourceImage = allImages.remove(0);
        Tile[] tiles = new Tile[allImages.size()];
        for (int i = 0; i < tiles.length; i++) {
            tiles[i] = new Tile(allImages.get(i));
        }
        Board board = new Board(sourceImage, tiles);
        return board.toArray();
    }
}

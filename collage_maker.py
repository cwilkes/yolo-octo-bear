import sys
from collections import defaultdict


def lgi(msg, *args):
    print >>sys.stderr, msg % args
    sys.stderr.flush()


class Board(object):
    def __init__(self, source_image, images):
        self.source_image = source_image
        self.tiles = list()
        for image in images:
            self.tiles.append(Tile(image))

    def remove(self, tile_pos):
        self.place(tile_pos, -1, -1, -1, -1)

    def place(self, tile_pos, top_col, top_row, width, height):
        tile = self.tiles[tile_pos]
        tile.top_col, tile.top_row = top_col, top_row
        tile.width, tile.height = width, height

    def as_ary(self):
        ret = list()
        for tile in self.tiles:
            ret.extend(tile.as_ary())
        return ret


class Tile(object):
    def __init__(self, image, top_col=-1, top_row=-1, width=-1, height=-1):
        self.image = image
        self.top_col, self.top_row, self.width, self.height = top_col, top_row, width, height
        self.cached_data = defaultdict(dict)

    def move(self, top_col, top_row):
        self.top_col, self.top_row = top_col, top_row

    def _is_off_board(self):
        return self.top_col == -1 or self.width == -1

    def data(self):
        if self._is_off_board():
            return None
        if self.width in self.cached_data and self.height in self.cached_data[self.width]:
            return self.cached_data[self.width][self.height]
        ret = list()
        ratio_w = 1.0 * self.image.width / self.width
        ratio_h = 1.0 * self.image.height / self.height
        ratio_w_int, ratio_h_int = int(ratio_w), int(ratio_h)
        top_r = 0
        for r1 in range(self.height):
            row = list()
            top_c = 0
            for c1 in range(self.width):
                count, color_sum = 0, 0
                for r2 in range(ratio_h_int):
                    for c2 in range(ratio_w_int):
                        try:
                            color_sum += self.image[int(top_r+r1)][int(top_c+c2)]
                            count += 1
                        except:
                            pass
                # should really do the next fraction
                if count > 0:
                    row.append(color_sum / count)
                else:
                    lgi('No nodes at (%d,%d) : (%d,%d) + (%d,%d)', c1, r1, top_r, top_c, c2, r2)
                top_c += ratio_w
            ret.append(row)
            top_r += ratio_h
        if ret:
            lgi('Size for %s is (%d,%d)', self.image, len(ret[0]), len(ret))
            self.cached_data[self.width][self.height] = ret
            return ret
        else:
            raise Exception('No data available for image %s with size (%d,%d)', self.image, self.width, self.height)

    def as_ary(self):
        if self._is_off_board():
            return [-1, -1, -1, -1]
        else:
            return [self.top_row, self.top_col, self.top_row+self.height, self.top_col+self.width]


class Image(object):
    def __init__(self, index, data):
        self.index = index
        self.data = data
        self.width, self.height = len(self.data[0]), len(self.data)

    def __getitem__(self, item):
        try:
            return self.data[item]
        except IndexError as ex:
            raise Exception('Cannot get item %d out of data of size %d : %s' % (item, len(self.data), ex))

    def __repr__(self):
        return '<image:%d %dx%d>' % (self.index, self.width, self.height)


def do_work(source_image, images):
    board = Board(source_image, images)
    num_nodes = 14
    width = source_image.width / num_nodes
    height = source_image.height / num_nodes
    row = 0
    index = 0
    end_width = source_image.width - num_nodes*width
    end_height = source_image.height - num_nodes*height
    lgi('Width: %d, end width: %d.  Height: %d, end height: %d', width, end_width, height, end_height)
    for r in range(14):
        col = 0
        for c in range(14):
            if r == 13:
                h = source_image.height - row-1
            else:
                h = height
            if c == 13:
                w = source_image.width - col-1
            else:
                w = width
            board.place(index, col, row, w, h)
            index += 1
            col += width + 1
        row += height + 1
    # now rescale all their image sizes
    for tile in board.tiles:
        tile.data()
    return board.as_ary()


def make_images(image_collection):
    images = list()
    index_pos = 0
    while index_pos < len(image_collection):
        height, width = image_collection[index_pos], image_collection[index_pos+1]
        index_pos += 2
        data = list()
        for r in range(height):
            row = list()
            for _ in range(width):
                row.append(image_collection[index_pos])
                index_pos += 1
            data.append(row)
        images.append(Image(len(images), data))
    return images.pop(0), images


def do_check(ret, images):
    board = list()
    for i in range(0,len(ret), 4):
        piece = i / 4 + 1
        top_row = ret[i]
        if top_row == -1:
            continue
        top_col = ret[i+1]
        bottom_row = ret[i+2]
        bottom_col = ret[i+3]
        for r in range(top_row, bottom_row+1):
            while r >= len(board):
                board.append(list())
            row = board[r]
            for c in range(top_col, bottom_col+1):
                while c >= len(row):
                    row.append(0)
                if row[c] != 0:
                    lgi('Error at (%d,%d) piece %d (%s) already exists, cannot put %d (%s) on it',
                        c, r, row[c], images[row[c]-1], piece, images[piece-1])
                row[c] = piece


def compose(image_collection):
    source_image, images = make_images(image_collection)
    lgi('Source image %s', source_image)
    ret = do_work(source_image, images)
    do_check(ret, images)
    return ret

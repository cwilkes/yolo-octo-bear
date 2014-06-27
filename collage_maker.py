import sys


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

    def as_ary(self):
        if self.top_col == -1:
            return [-1, -1, -1, -1]
        else:
            return [self.top_row, self.top_col, self.top_row+self.height, self.top_col+self.width]


class Image(object):
    def __init__(self, index, data):
        self.index = index
        self.data = data
        self.width, self.height = len(self.data[0]), len(self.data)

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

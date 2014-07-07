from util import lgi
import datetime
import math
from collections import defaultdict, namedtuple


SCORE_UNPLACED = 200000
DIM = namedtuple('DIM', 'width height')
COORD = namedtuple('COORD', 'row col')


def get_rows(coord, dim): return range(coord.row, coord.row+dim.height)


def get_cols(coord, dim): return range(coord.col, coord.col + dim.width)


def get_row_cols(coord, dim):
    for r in get_rows(coord, dim):
        for c in get_cols(coord, dim):
            yield r, c


class Board(object):
    def __init__(self, source_image, images):
        self.source_image = source_image
        self.source_breaks = dict()
        self.tiles = list()
        for image in images:
            self.tiles.append(Tile(image))
        lgi('First tile out of %d: %r', len(self.tiles), self.tiles[0])
        if source_image.width == 300:
            # 300/20 = 15 pixels
            self.number_width = 30
            self.number_height = 15
        else:
            self.number_width = 15
            self.number_height = 30
        self.width_size, self.height_size = source_image.width / self.number_width, source_image.height / self.number_height
        self.source_image_blocks = self.source_image.average_blocks(self.number_width, self.number_height)
        self.placed = list()
        for r in self.number_height:
            self.placed.append([-1 for _ in range(self.number_width)])
        self.score_cache = dict()
        self.total_score = 0
        self.max_slots = list()
        for img in images:
            self.max_slots.append(DIM(img.width/self.width_size, img.height/self.height_size))

    def score(self, tile_pos, coord, dim):
        if coord.row == -1:
            return SCORE_UNPLACED
        key = tile_pos, coord, dim
        if key in self.score_cache:
            return self.score_cache[key]
        blocks = self.tiles[tile_pos]._image.average_blocks(dim.width, dim.height)
        t = 0
        for r in get_rows(coord, dim):
            for c in get_cols(coord, dim):
                t += math.pow(self.source_image_blocks[r][c]-blocks[r][c], 2)
        t = math.sqrt(t)
        self.score_cache[key] = t
        return t

    def swap(self, tile_pos1, tile_pos2):
        tile1, tile2 = self.tiles[tile_pos1], self.tiles[tile_pos2]
        tile1_pos, tile2_pos = tile1.rcwh(), tile2.rcwh()

        score1_pre = self.score(tile_pos1, tile1_pos[0], tile1_pos[1], tile1_pos[2], tile1_pos[3])
        score2_pre = self.score(tile_pos2, tile2_pos[0], tile2_pos[1], tile2_pos[2], tile2_pos[3])

        self.place(tile_pos1, tile2_pos[0], tile2_pos[1], tile2_pos[2], tile2_pos[3], throw_exception=False)
        self.place(tile_pos2, tile1_pos[0], tile1_pos[1], tile1_pos[2], tile1_pos[3], throw_exception=False)

        score1_post = self.score(tile_pos1, tile1_pos[0], tile1_pos[1], tile1_pos[2], tile1_pos[3])
        score2_post = self.score(tile_pos2, tile2_pos[0], tile2_pos[1], tile2_pos[2], tile2_pos[3])

        if score1_post+score2_post < score1_pre+score2_pre:
            # keep it
            self.total_score -= score1_pre+score2_pre
            self.total_score += score1_post+score2_post
        else:
            # revert
            self.place(tile_pos1, tile1_pos[0], tile1_pos[1], tile1_pos[2], tile1_pos[3], throw_exception=False)
            self.place(tile_pos2, tile2_pos[0], tile2_pos[1], tile2_pos[2], tile2_pos[3], throw_exception=False)

    def hits(self, row_slot, col_slot, width_slots, height_slots):
        seen = set()
        for r in range(row_slot, row_slot+height_slots):
            for c in range(col_slot, col_slot+width_slots):
                v = self.places[r][c]
                if v != -1 and not v in seen:
                    seen.add(v)
                    yield v

    def remove(self, tile_pos):
        tile = self.tiles[tile_pos]
        r, c, w, h = tile.rcwh()
        if r != -1:
            sc = self.score_cache[(tile_pos, r, c, w, h)]
            self.total_score -= sc
            tile.move(-1, -1)
            tile.resize(-1, -1)
            for y in range(r, r+h):
                for x in range(c, c+w):
                    self.placed[y][x] = -1
        return r, c, w, h

    def place(self, tile_pos, row_slot, col_slot, width_slots, height_slots, log=False, do_check=True, throw_exception=True):
        existing = list(self.hits(row_slot, col_slot, width_slots, height_slots))
        if throw_exception and existing:
            raise Exception('Already has these blocks: %r' % (existing, ))
        # now remove those existing, if any
        replace = [self.remove(_) for _ in existing]
        sc = self.score(tile_pos, row_slot, col_slot, width_slots, height_slots)
        self.total_score += sc
        for y in range(row_slot, row_slot + height_slots):
            for x in range(col_slot, col_slot + width_slots):
                self.placed[y][x] = tile_pos
        self.tiles[tile_pos].move(row_slot, col_slot)
        self.tiles[tile_pos].resize(width_slots, height_slots)
        if log:
            lgi('Placed img %r at (%d,%d) with size (%d,%d).  Score: %d, Total score: %d',
                tile_pos, row_slot, col_slot, width_slots, height_slots, sc, self.total_score)
        return replace

    def as_ary(self):
        ret = list()
        for tile in self.tiles:
            if tile.placed():
                start_row = tile._top_row * self.height_size
                start_col = tile._top_col * self.width_size
                if tile._top_col + tile._width == self.number_width:
                    end_col = self.source_image.width-1
                else:
                    end_col = start_col + tile._width * self.width_size
                if tile._top_row + tile._height == self.number_height:
                    end_row = self.source_image.height-1
                else:
                    end_row = start_row + tile._height * self.height_size
                ret.extend([start_row, start_col, end_row, end_col])
            else:
                ret.extend([-1, -1, -1, -1])
        return ret


class Tile(object):
    """An image with a position and a scaled size
    """
    def __init__(self, image, coord=None, dim=None):
        self.index = image.index
        self._image = image
        self._coord, self._dim = coord, dim
        self.resize(self._dim)

    def move(self, top_row, top_col):
        self._top_col, self._top_row = top_col, top_row

    def resize(self, dim):
        if not dim:
            self._dim = None
            return self._dim
        if dim.width < 0:
            raise Exception('Width should not be less than 0: %d' % (dim.width, ))
        if dim.height < 0:
            raise Exception('Height should not be less than 0: %d' % (dim.height, ))
        if dim.width >= self._image.width:
            lgi('Asked to make width larger (%d) than original width (%d).  Setting to max', dim.width, self._image.width)
            w = self._image.width
        else:
            w = dim.width
        if dim.height >= self._image.height:
            lgi('Asked to make height larger (%d) than original height (%d).  Setting to max', dim.height, self._image.height)
            h = self._image.height
        else:
            h = dim.height
        self._dim = DIM(w, h)
        return self._dim

    def placement(self):
        return self._coord, self._dim

    def placed(self):
        return self._coord and self._dim

    def __repr__(self):
        return '<Tile: %d %r coord %s size %s>' % (self.index, self._image, self._coord, self._dim)


class Image(object):
    """image colors are [0,255] inclusive
    """
    def __init__(self, index, data):
        if type(index) is not int:
            raise Exception('Index (%r) should be an int, not %s' % (index, type(index)))
        self.index = index
        self._data = data
        self._dim = DIM(len(self._data[0]), len(self._data))
        self.avg_cache = dict()

    @property
    def width(self):
        return self._dim.width

    @property
    def height(self):
        return self._dim.height

    def __getitem__(self, item):
        try:
            return self._data[item]
        except IndexError as ex:
            raise Exception('Cannot get item %d out of data of size %d : %s' % (item, len(self.data), ex))

    def __len__(self):
        return self._dim.height

    def __repr__(self):
        return '<image:%s %dx%d>' % (self.index, self._dim.width, self._dim.height)

    def _get_rows(self, number_height):
        rows = list()
        r = 0
        block_height = 1.0 * self.height / number_height
        for _ in range(number_height):
            next_val = min(r+block_height, self.height-1)
            r2 = int(r)
            rows.append((r2, next_val-r2))
            r += next_val
        return rows

    def _get_cols(self, number_width):
        cols = list()
        c = 0
        block_width = 1.0 * self.width / number_width
        for _ in range(number_width):
            next_val = min(c+block_width, self.width-1)
            c2 = int(c)
            cols.append((c2, next_val-c2))
            c += next_val
        return cols

    def average_blocks(self, dim):
        if dim in self.avg_cache:
            return self.avg_cache[dim]
        block_height = 1.0 * self.height / dim.height
        block_width = 1.0 * self.width / dim.width
        for r_pos in range(dim.height):
            r = r_pos * block_height
            for c_pos in range(dim.width):



        rows = self._get_rows(number_height)
        cols = self._get_cols(number_width)
        ret = list()
        for r, h in rows:
            by_row = list()
            for c, w in cols:
                by_row.append(self.average(r, c, w, h))
            ret.append(by_row)
        self.avg_cache[dim] = ret
        return ret

    def average(self, offset_row, offset_col, width, height):
        key = offset_row, offset_col, width, height
        if key in self.avg_cache:
            return self.avg_cache[key]
        tot = 0
        for row in self._image._data[offset_row:offset_row+height]:
            tot += sum(row[offset_col:offset_col+width])
        avg = tot / (width*height)
        self.avg_cache[key] = avg
        return avg


class MyTimer(object):
    def __init__(self, max_millis):
        self.start_time = datetime.datetime.now()
        self.max_millis = max_millis

    def delta(self):
        dt = datetime.datetime.now() - self.start_time
        return 1000*1000*dt.seconds + dt.microseconds

    def should_finish(self):
        return self.delta() > self.max_millis
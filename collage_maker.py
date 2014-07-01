import sys
from collections import defaultdict, Counter
import math
import random
import datetime


random.seed(1)

SCORE_UNPLACED = 2
MAX_MILLIS = 8 * 1000 * 1000


def lgi(msg, *args):
    print >>sys.stderr, msg % args
    sys.stderr.flush()


def get_slice(data, col_offset, row_offset, width, height):
    ret = list()
    for r in range(height):
        row = list()
        for c in range(width):
            row.append(data[row_offset+r][col_offset+c])
        ret.append(row)
    return ret


class Board(object):
    def __init__(self, source_image, images):
        self.source_image = source_image
        self.source_breaks = dict()
        self.tiles = list()
        for image in images:
            self.tiles.append(Tile(image))
        lgi('First tile out of %d: %r', len(self.tiles), self.tiles[0])

    def threshold_scoring(self, width, height, thresholds, offset_col=0, offset_row=0):
        if type(thresholds) == int:
            thresholds = [thresholds, ]
        ret = list()
        for row in range(offset_row, self.source_image.height, height):
            if row + height >= self.source_image.height:
                continue
            lgi('Row %d => %d', row, row+height)
            for col in range(offset_col, self.source_image.width, width):
                if col + width >= self.source_image.width:
                    continue
                key = '%d,%d:%d,%d' % (col, row, width, height)
                if not key in self.source_breaks:
                    self.source_breaks[key] = Image(10000+row*self.source_image.width+col, get_slice(self.source_image, col, row, width, height))
                img = self.source_breaks[key]
                scores = img.threshold_scoring(thresholds)
                ret.append((col, row, scores))
        return ret

    def score(self, tile_pos):
        tile = self.tiles[tile_pos]
        data = tile.data()
        if data is None:
            return -1
        s = 0
        offset_row, offset_col = tile._top_row, tile._top_col
        for r in range(len(data)):
            row = data[r]
            for c, val in enumerate(row):
                try:
                    other_val = self.source_image[offset_row+r][offset_col+c]
                    s += (other_val-val)*(other_val-val)
                except IndexError as ex:
                    lgi('Error with tile %s offset (%d,%d) add (%d,%d) => (%d,%d) : %s',
                        self.tiles[tile_pos], offset_col, offset_row, c, r, offset_col+c, offset_row+r, ex)
        return s

    def swap(self, tile_pos1, tile_pos2):
        tile1, tile2 = self.tiles[tile_pos1], self.tiles[tile_pos2]
        tile1_pos, tile1_dim = tile1.coord(), tile1.dim()
        tile2_pos, tile2_dim = tile2.coord(), tile2.dim()
        self.place(tile_pos1, tile2_pos[0], tile2_pos[1], tile2_dim[0], tile2_dim[1])
        self.place(tile_pos2, tile1_pos[0], tile1_pos[1], tile1_dim[0], tile1_dim[1])

    def place(self, tile_pos, top_row, top_col, width=None, height=None, log=False, do_check=True):
        try:
            tile = self.tiles[tile_pos]
        except:
            raise Exception('Asking for tile_pos %r in array size %r' % (tile_pos, len(self.tiles)))
        prev = repr(tile)
        tile.move(top_row, top_col)
        if width is not None:
            tile.resize(width, height)
        if log:
            lgi('Changed tile %r to %r', prev, tile)
        if do_check:
            box = tile.box()
            if box[0] != -1:
                if box[0] < 0 or box[1] < 0:
                    raise Exception('%r is above or to right of image %r' % (tile, self.source_image))
                if box[2] >= self.source_image.height:
                    raise Exception('%r is below %r' % (tile, self.source_image))
                if box[3] >= self.source_image.width:
                    raise Exception('%r is to right of %r' % (tile, self.source_image))
        return tile.dim()

    def as_ary(self):
        ret = list()
        for tile in self.tiles:
            ret.extend(tile.box())
        return ret

    def dim(self, tile_id):
        return self.tiles[tile_id].dim()

    def coord(self, tile_id):
        return self.tiles[tile_id].coord()

    def row_count(self):
        # should make more efficient
        c = set()
        for tile in self.tiles:
            c.add(tile._top_row)
        return len(c)


class Tile(object):
    """An image with a position and a scaled size
    """
    def __init__(self, image, top_row=-1, top_col=-1, width=-1, height=-1):
        self.index = image.index
        self._image = image
        self._top_col, self._top_row = top_col, top_row
        self._width, self._height = width, height
        self.resize(width, height)

    def move(self, top_row, top_col):
        self._top_col, self._top_row = top_col, top_row

    def resize(self, width, height):
        if width < -1:
            raise Exception('Width should not be less than -1: %d' % (width, ))
        if height < -1:
            raise Exception('Height should not be less than -1: %d' % (height, ))
        if width >= self._image.width:
            lgi('Asked to make width larger (%d) than original width (%d).  Setting to max', width, self._image.width)
            self._width = self._image.width
        else:
            self._width = width
        if height >= self._image.height:
            lgi('Asked to make height larger (%d) than original height (%d).  Setting to max', height, self._image.height)
            self._height = self._image.height
        else:
            self._height = height
        return self._width, self._height

    def placed(self):
        return self._width != -1 and self._top_col != -1

    def box(self):
        if self.placed():
            return self._top_row, self._top_col, self._top_row+self._height-1, self._top_col+self._width-1
        else:
            return -1, -1, -1, -1

    def dim(self):
        return self._width, self._height

    def coord(self):
        return self._top_row, self._top_col

    def __repr__(self):
        return '<Tile: %d %r dim %r size %r>' % (self.index, self._image, self.box(), self.dim())


class Image(object):
    """image colors are [0,255] inclusive
    """
    def __init__(self, index, data, sub_title=None):
        if type(index) is not int:
            raise Exception('Index (%r) should be an int, not %s' % (index, type(index)))
        self.index = index
        self._data = data
        self.width, self.height = len(self._data[0]), len(self._data)
        self.sub_title = sub_title

    def __getitem__(self, item):
        try:
            return self._data[item]
        except IndexError as ex:
            raise Exception('Cannot get item %d out of data of size %d : %s' % (item, len(self.data), ex))

    def __len__(self):
        return self.height

    def __repr__(self):
        if self.sub_title is None:
            img_id = self.index
        else:
            img_id = '%s_%s' % (self.index, self.sub_title)
        return '<image:%s %dx%d>' % (img_id, self.width, self.height)


def _get_min_width_and_height(images):
    min_width, min_height = sys.maxint, sys.maxint
    for img in images:
        min_width = min(min_width, img.width)
        min_height = min(min_height, img.height)
    return min_width, min_height


def find_matches(board_scores, scaled_images_scores):
    tiles_to_images = list()
    used_tiles = set()
    for r, c, s in board_scores:
        best_img_index, best_img_score = None, sys.maxint
        for img_index, img_scores in scaled_images_scores:
            if img_index in used_tiles:
                continue
            my_score = sum([(a-b)*(a-b) for a,b in zip(img_scores, s)])
            if my_score < best_img_score:
                best_img_index, best_img_score = img_index, my_score
        if best_img_index is None:
            break
        tiles_to_images.append((r, c, best_img_index))
        used_tiles.add(best_img_index)
    return tiles_to_images


def fill_sides(board, free_tiles, ending_col, ending_row):
    row = 0
    lgi('Filling in right col %r', ending_col)
    source_image = board.source_image
    while row < source_image.height:
        tile = free_tiles.pop(0)
        img = tile._image
        h = min(source_image.height, board.source_image.height-row)
        w = source_image.width - ending_col
        w, h = board.place(tile.index, row, ending_col, w, h, True)
        row += h
    col = 0
    lgi('Filling in bottom row %r', ending_row)
    while col < ending_col:
        tile = free_tiles.pop(0)
        img = tile._image
        w = min(img.width, ending_col-col)
        h = source_image.height-ending_row
        if w <= 0 or h <= 0:
            break
        board.place(tile.index, ending_row, col, w, h, True)
        col += w


def image_histogram(data, offsets=None):
    c = [0 for _ in range(256)]
    if offsets:
        row_start, col_start, row_end, col_end = offsets
    else:
        row_start, row_end = 0, len(data)
        col_start, col_end = 0, len(data[0])
    for row in data[row_start:row_end]:
        for val in row[col_start:col_end]:
            c[val] += 1
    side_buffers = list()
    img_size = (row_end-row_start)*(col_end-col_start)
    for pos in range(256):
        val = 0
        if pos > 1:
            val += 0.25 * c[pos-2]
        if pos > 0:
            val += 0.5 * c[pos-1]
        val += c[pos]
        if pos <= 254:
            val += 0.5 * c[pos+1]
        if pos <= 253:
            val += 0.25 * c[pos+2]
        side_buffers.append(1.0 * val / img_size)
    return side_buffers


def histo_large_image(data, width, height, overall_histo):
    info = dict()
    o_o = [_[0] for _ in overall_histo]
    for row in range(0, len(data)-height, height):
        for col in range(0, len(data[0])-width, width):
            hi = image_histo_offsets(data, offsets=(col, col+width, row, row+height))
            #lgi('At (%d,%d) histo: %r', col, row, list(hi.most_common()))
            cume = 0
            ret = list()
            my_colors = sorted(hi.keys())
            tot = 0
            for top_end in o_o:
                #lgi('Working on top end %r with %r', top_end, my_colors)
                while my_colors and my_colors[0] <= top_end:
                    cume += hi[my_colors.pop(0)]
                ret.append(cume)
                tot += cume
                cume = 0
            if cume != 0:
                raise Exception('Cume should be zero not %d' % (cume, ))
            info[(col, row)] = [(100*_+50) / tot for _ in ret]
            lgi('Histo at (%d,%d) : %r', col, row, info[(col, row)])
    return info


def _get_random_image_ids(number):
    ids = list(range(0, number))
    random.shuffle(ids)
    return ids


class MyTimer(object):
    def __init__(self):
        self.start_time = datetime.datetime.now()

    def delta(self):
        dt = datetime.datetime.now() - self.start_time
        return 1000*1000*dt.seconds + dt.microseconds

    def should_finish(self):
        return self.delta() > MAX_MILLIS


def bucketized_histogram(histogram, breaks):
    ret = list()
    cume = 0
    b = [_[0] for _ in breaks]
    stop = b.pop(0)
    for pos, val in enumerate(histogram):
        cume += val
        if pos == stop:
            ret.append(cume)
            cume = 0
            if not b:
                break
            stop = b.pop(0)
    return ret


class ImageClassifier(object):
    def __init__(self, board, width, height, timer):
        self.board = board
        self.width, self.height = width, height
        self.image_board_score = defaultdict(dict)
        self.total_score = 0
        self._source_image_histogram = dict()
        self.number_images = len(board.tiles)
        self.placed_coords = list()
        self.total_score = 0
        self.timer = timer
        for row in range(0, len(board.source_image)-height, height):
            for col in range(0, len(board.source_image[0])-width, width):
                self.placed_coords.append((row, col))
        lgi('Computing histograms')
        self._image_histograms = list()
        for tile in board.tiles:
            self._image_histograms.append(image_histogram(tile._image._data))
        lgi('Done computing histograms')
        self.histogram_breaks = find_histo_breaks(self._image_histograms, 8)
        lgi('Histo breaks: %r', self.histogram_breaks)
        self.image_histograms_breaks = list()
        self.source_image_histogram_breaks = dict()
        for histo in self._image_histograms:
            self.image_histograms_breaks.append(bucketized_histogram(histo, self.histogram_breaks))
        for row, col in self.placed_coords:
            raw = image_histogram(board.source_image, offsets=(row, col, row+height, col+height))
            self.source_image_histogram_breaks[(row, col)] = bucketized_histogram(raw, self.histogram_breaks)

    def _get_score(self, img_id):
        key = self.board.coord(img_id)
        if key[0] == -1:
            return SCORE_UNPLACED
        if key in self.image_board_score:
            if img_id in self.image_board_score[key]:
                return self.image_board_score[key][img_id]
        overall_histogram = self.source_image_histogram_breaks[key]
        tot = 0
        for a, b in zip(overall_histogram, self.image_histograms_breaks[img_id]):
            tot += math.pow(a-b, 2)
        tot = math.sqrt(tot)
        self.image_board_score[key][img_id] = tot
        return tot

    def _check_swap(self, img1, img2):
        score1_pre, score2_pre = self._get_score(img1), self._get_score(img2)
        img1_coord, img2_coord = self.board.coord(img1), self.board.coord(img2)
        self.board.swap(img1, img2)
        score1_post, score2_post = self._get_score(img1), self._get_score(img2)
        delta_score = (score1_pre + score2_pre) - (score1_post + score2_post)
        #lgi('Swapped ids %r and %r, pre: (%r,%r), post: (%r,%r) => %r', img1, img2, score1_pre, score2_pre, score1_post, score2_post, delta_score)
        if delta_score > 0:
            # good move, adjust total score
            self.total_score -= delta_score
            return True
        else:
            # revert
            #lgi('Bad move with ids %r and %r, pre: (%d,%d), post: (%d,%d) => %r', img1, img2, score1_pre, score2_pre, score1_post, score2_post, delta_score)
            self.board.swap(img1, img2)
            self.board.place(img1, img1_coord[0], img1_coord[1])
            self.board.place(img2, img2_coord[0], img2_coord[1])
            return False

    def match(self):
        x = list(self.placed_coords)
        for img_index in _get_random_image_ids(self.number_images):
            row, col = x.pop(0)
            self.board.place(img_index, row, col, self.width, self.height, log=True)
            self.total_score += self._get_score(img_index)
            if not x:
                break
        lgi('initial placement score: %r', self.total_score)
        for t in range(10):
            number_swaps = 0
            for id1 in range(self.number_images):
                for id2 in range(self.number_images):
                    if id1 == id2:
                        continue
                    if self.timer.should_finish():
                        break
                    if self._check_swap(id1, id2):
                        number_swaps += 1
                        break
                if self.timer.should_finish():
                    break
            lgi('Score in round %r after %d swaps: %r', t, number_swaps, self.total_score)
            if self.timer.should_finish():
                lgi('Breaking due to time')
                break
            if number_swaps <= 1:
                break
        lgi('Final score: %r', self.total_score)
        lgi('Total time: %r', self.timer.delta() / 1000)


def get_free_tiles(board):
    free_tiles = list()
    ending_col, ending_row = 0, 0
    for tile in board.tiles:
        box = tile.box()
        if box[0] == -1:
            free_tiles.append(tile)
        else:
            ending_col = max(ending_col, box[3]+1)
            ending_row = max(ending_row, box[2]+1)
    return free_tiles, ending_col, ending_row


def find_histo_breaks(img_histos, number_breaks):
    all_colors = [0 for _ in range(256)]
    total_count = 0
    for histo in img_histos:
        for pos, val in enumerate(histo):
            all_colors[pos] += val
            total_count += val
    delta = 1.0 * total_count / number_breaks
    cume = 0
    next_val = delta
    breaks = list()
    prev = 0
    for color in range(256):
        cume += all_colors[color]
        if cume >= next_val:
            breaks.append((color, cume-prev))
            next_val += delta
            prev = cume
    breaks.append((color, cume-prev))
    if color != 255:
        breaks.append((255, total_count-cume))
    return breaks


def do_work(source_image, images, timer):
    #overall_histo, per_image_histo = bucketize_histograms(images)
    #lgi('x: %r', overall_histo)
    board = Board(source_image, images)
    min_width, min_height = _get_min_width_and_height(images)
    min_width = 2 * (min_width / 2)
    min_height = 2 * (min_height / 2)
    min_width, min_height = int(min_width / 1.5), int(min_height / 1.5)
    lgi('Min width, height: (%d,%d)', min_width, min_height)
    #scaled_images = [board.place(tile.index, -1, -1, min_width, min_height) for tile in board.tiles]
    ic = ImageClassifier(board, min_width, min_height, timer)
    #ic.match()
    #ic.match2(overall_histo, per_image_histo, min_width, min_height)
    ic.match()
    free_tiles, ending_col, ending_row = get_free_tiles(board)
    fill_sides(board, free_tiles, ending_col, ending_row)
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
        images.append(Image(len(images)-1, data))
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
                    lgi('Error at (%d,%d) piece %s already exists, cannot put %s on it',
                        c, r, images[row[c]-1], images[piece-1])
                row[c] = piece


class CollageMaker(object):
    def __init__(self):
        pass

    def compose(self, image_collection):
        timer = MyTimer()
        source_image, images = make_images(image_collection)
        lgi('Source image %s', source_image)
        ret = do_work(source_image, images, timer)
        lgi('Total time: %r', timer.delta()/1000)
        #do_check(ret, images)
        return ret

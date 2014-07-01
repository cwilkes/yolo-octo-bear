import sys
from collections import defaultdict, Counter
import math
import random
import datetime


random.seed(1)

SCORE_UNPLACED = 200000
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
            self.tiles.append(Tile(len(self.tiles), image))

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

    def remove(self, tile_pos):
        self.place(tile_pos, -1, -1, -1, -1)

    def place(self, tile_pos, top_col, top_row, width=None, height=None, log=False, do_check=True):
        try:
            tile = self.tiles[tile_pos]
        except:
            raise Exception('Asking for tile_pos %r in array size %r' % (tile_pos, len(self.tiles)))
        prev = repr(tile)
        tile.move(top_col, top_row)
        if width is not None:
            tile.resize(width, height)
        if log:
            lgi('Changed tile %r to %r', prev, tile)
        if do_check:
            box = tile.box()
            if box[0] is not None:
                if box[0] < 0 or box[1] < 0:
                    raise Exception('Tile %r box %r is above or to right of image %r' % (tile, box, self.source_image))
                if box[2] >= self.source_image.width or box[3] >= self.source_image.height:
                    raise Exception('Tile %r box %r is below or to right of image %r' % (tile, box, self.source_image))
        return tile._mini_image

    def as_ary(self):
        ret = list()
        for tile in self.tiles:
            ret.extend(tile.as_ary())
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
    def __init__(self, index, image, top_col=-1, top_row=-1, width=-1, height=-1):
        self.index = index
        self._image = image
        self._top_col, self._top_row = top_col, top_row
        self._mini_image = None
        self.resize(width, height)

    def move(self, top_col, top_row):
        self._top_col, self._top_row = top_col, top_row

    def resize(self, width, height):
        if width < -1:
            raise Exception('Width should not be less than -1: %d' % (width, ))
        if height < -1:
            raise Exception('Height should not be less than -1: %d' % (height, ))
        if width == -1 or height == -1:
            self._mini_image = None
        else:
            self._mini_image = self._image.scale(width, height)

    def placed(self):
        return self._mini_image is not None and self._top_col != -1

    def box(self):
        if self.placed():
            return self._top_col, self._top_row, self._top_col + self._mini_image.width-1, self._top_row + self._mini_image.height-1
        else:
            return None, None, None, None

    def data(self):
        if self._mini_image:
            return self._mini_image.data
        else:
            return None

    def dim(self):
        if self._mini_image is None:
            return None, None
        else:
            return self._mini_image.width, self._mini_image.height

    def coord(self):
        return self._top_col, self._top_row

    def __repr__(self):
        if self._mini_image:
            return '<Tile: @ (%d,%d-%d,%d) image %s : %s>' % \
                   (self._top_col, self._top_row, self._top_col + self._mini_image.width-1, self._top_row + self._mini_image.height-1, self._image, self._mini_image)
        else:
            return '<Tile: @ (%d,%d-NA) image %s : %s>' % (self._top_col, self._top_row, self._image, self._mini_image)

    def as_ary(self):
        if self.placed():
            return [self._top_row, self._top_col, self._top_row+self._mini_image.height-1, self._top_col+self._mini_image.width-1]
        else:
            return [-1, -1, -1, -1]


class Image(object):
    def __init__(self, index, data, sub_title=None):
        if type(index) is not int:
            raise Exception('Index (%r) should be an int, not %s' % (index, type(index)))
        self.index = index
        self.data = data
        self.width, self.height = len(self.data[0]), len(self.data)
        self.sub_title = sub_title

    def threshold_scoring(self, thresholds):
        t2 = thresholds + [255]
        counts = [0 for _ in range(len(thresholds)+1)]
        for r2 in range(self.height):
            for c2 in range(self.width):
                for pos, val in enumerate(t2):
                    if self.data[r2][c2] <= val:
                        counts[pos] += 1
                        break
        return counts

    def scale(self, width, height):
        ret = list()
        ratio_w = 1.0 * self.width / width
        ratio_h = 1.0 * self.height / height
        if ratio_w < 1.0 or ratio_h < 1.0:
            raise Exception('Resizing image %s should be downsized, not going to (%d,%d) (ratios: %r, %r)' % (self, width, height, ratio_w, ratio_h))
        ratio_w_int, ratio_h_int = int(ratio_w), int(ratio_h)
        top_r = 0
        for r1 in range(height):
            row = list()
            top_c = 0
            for c1 in range(width):
                count, color_sum = 0, 0
                for r2 in range(ratio_h_int):
                    for c2 in range(ratio_w_int):
                        try:
                            color_sum += self.data[int(top_r+r2)][int(top_c+c2)]
                            count += 1
                        except Exception as ex:
                            lgi('For image %s at (%d,%d) have no nodes at (%d,%d) + (%d,%d) size (%d,%d): %s', self, top_c, top_r, c1, r1, c2, r2, len(self.data), len(self.data[0]), ex)
                # should really do the next fraction
                if count > 0:
                    row.append(color_sum / count)
                else:
                    lgi('For image %s at (%d,%d) have no nodes at (%d,%d) ratios (%r,%r)',
                        self._image, top_c, top_r, c1, r1, ratio_w, ratio_h)
                top_c += ratio_w
            ret.append(row)
            top_r += ratio_h
        return Image(self.index, ret, '%r_%r' % (width, height))

    def __getitem__(self, item):
        try:
            return self.data[item]
        except IndexError as ex:
            raise Exception('Cannot get item %d out of data of size %d : %s' % (item, len(self.data), ex))

    def __sizeof__(self):
        return len(self.data)

    def __repr__(self):
        if self.sub_title is None:
            img_id = self.index
        else:
            img_id = '%s_%s' % (self.index, self.sub_title)
        return '<image:%r %dx%d>' % (img_id, self.width, self.height)


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
        w = source_image.width-ending_col
        h = min(img.height, source_image.height-row)
        board.place(tile._image.index, ending_col, row, w, h, True)
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
        board.place(tile._image.index, col, ending_row, w, h, True)
        col += w


def image_histo_around(img):
    normal = image_histo(img.data)
    half_size = Counter()
    for r in range(0, len(img.data)-1, 2):
        for c in range(0, len(img.data[0])-1, 2):
            col = img.data[r][c]
            col += img.data[r][c+1]
            col += img.data[r+1][c]
            col += img.data[r+1][c+1]
            half_size[col/4] += 1
    for color in half_size.keys():
        half_size[color] = 40000 * c[color] / (len(img.data)*len(img.data[0]))
    return normal, half_size


def image_histo(data):
    c = Counter()
    for row in data:
        for val in row:
            c[val] += 1
    for color in c.keys():
        c[color] = 10000 * c[color] / (len(data)*len(data[0]))
    return c


def image_histo_offsets(data, offsets):
    c = Counter()
    for row in data[offsets[2]:offsets[3]]:
        for val in row[offsets[0]:offsets[1]]:
            c[val] += 1
    for color in c.keys():
        c[color] = 10000 * c[color] / (len(data)*len(data[0]))
    return c


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


class ImageClassifier(object):
    def __init__(self, board, source_image, width, height, start_time):
        self.start_time = start_time
        self.board = board
        self.source_image = source_image
        self.width, self.height = width, height
        self.image_board_score = defaultdict(dict)
        self.total_score = 0

    def delta(self):
        dt = datetime.datetime.now() - self.start_time
        return 1000*1000*dt.seconds + dt.microseconds

    def _get_score(self, img_id):
        dim1_col, dim1_row = self.board.coord(img_id)
        if dim1_col == -1:
            return SCORE_UNPLACED
        else:
            return self.image_board_score[(dim1_col, dim1_row)][img_id]

    def _check_swap(self, img1, img2):
        score1_pre, score2_pre = self._get_score(img1), self._get_score(img2)
        self.board.swap(img1, img2)
        score1_post, score2_post = self._get_score(img1), self._get_score(img2)
        delta_score = (score1_pre + score2_pre) - (score1_post + score2_post)
        if delta_score > 0:
            # good move, adjust total score
            lgi('Good move with ids %r and %r, pre: (%d,%d), post: (%d,%d) => %d', img1, img2, score1_pre, score2_pre, score1_post, score2_post, delta_score)
            self.total_score -= delta_score
        else:
            # revert
            self.board.swap(img1, img2)

    def match2(self, overall_histo, per_image_histo, width, height):
        sic = histo_large_image(self.source_image.data, self.width, self.height, overall_histo)
        locales = list(sic.keys())
        for img_index, histo in enumerate(per_image_histo):
            for key, bank in sic.items():
                tot = 0
                for a, b in zip(histo, bank):
                    tot += (a-b)**2
                self.image_board_score[key][img_index] = tot
            if locales:
                placed = locales.pop(0)
                self.board.place(img_index, placed[0], placed[1], width, height, log=True)
                self.total_score += tot
        lgi('Initial score: %r', self.total_score)
        for t in range(30):
            ids = _get_random_image_ids(len(per_image_histo))
            cur_score = self.total_score
            while len(ids) >= 2:
                delta_time = self.delta()
                if delta_time > MAX_MILLIS:
                    lgi('Breaking as time: %r', delta_time)
                    break
                id1, id2 = ids.pop(0), ids.pop(0)
                self._check_swap(id1, id2)
            lgi('Score at round %d: %r', t, self.total_score)
            if cur_score == self.total_score:
                break
        lgi('Final score: %r', self.total_score)
        lgi('Total time: %r', self.delta() / 1000)


def get_free_tiles(board):
    free_tiles = list()
    ending_col, ending_row = 0, 0
    for tile in board.tiles:
        box = tile.box()
        if box[0] is None:
            free_tiles.append(tile)
        else:
            ending_col = max(ending_col, box[2]+1)
            ending_row = max(ending_row, box[3]+1)
    return free_tiles, ending_col, ending_row


def bucketize_histograms(images):
    ret1 = list()
    all_colors = Counter()
    for img in images:
        c = image_histo(img.data)
        ret1.append(c)
        all_colors += c
    total_count = 0
    for color, count in all_colors.items():
        total_count += count
    delta = total_count / 8
    ret = list()
    cume = 0
    for color in sorted(all_colors.keys()):
        cume += all_colors[color]
        if cume >= delta:
            ret.append((color, cume))
            cume = 0
    ret.append((color, cume))
    if color != 255:
        ret.append((255, cume))
    per_image = list()
    for c in ret1:
        my_colors = sorted(c.keys())
        foo = list()
        for color in (_[0] for _ in ret):
            cur = 0
            while my_colors and my_colors[0] <= color:
                cur += c[my_colors.pop(0)]
            foo.append((cur+50)/100)
        per_image.append(foo)
    return ret, per_image


def do_work(source_image, images):
    start_time = datetime.datetime.now()
    overall_histo, per_image_histo = bucketize_histograms(images)
    lgi('x: %r', overall_histo)
    board = Board(source_image, images)
    min_width, min_height = _get_min_width_and_height(images)
    min_width = 2 * (min_width / 2)
    min_height = 2 * (min_height / 2)
    min_width, min_height = int(min_width / 1.5), int(min_height / 1.5)
    lgi('Min width, height: (%d,%d)', min_width, min_height)
    #scaled_images = [board.place(tile.index, -1, -1, min_width, min_height) for tile in board.tiles]
    ic = ImageClassifier(board, source_image, min_width, min_height, start_time)
    #ic.match()
    ic.match2(overall_histo, per_image_histo, min_width, min_height)
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
        source_image, images = make_images(image_collection)
        lgi('Source image %s', source_image)
        ret = do_work(source_image, images)
        #do_check(ret, images)
        return ret

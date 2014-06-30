import sys
from collections import defaultdict, Counter
import math
import random

random.seed(1)

SCORE_UNPLACED = 200000


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


def simulated_annealing(board, dist=5):
    for tile_pos1 in range(len(board.tiles)):
        if dist > 0:
            tile_pos2 = (tile_pos1 + dist) % len(board.tiles)
        else:
            tile_pos2 = (tile_pos1 - board.row_count() * dist) % len(board.tiles)
        t1, t2 = board.tiles[tile_pos1], board.tiles[tile_pos2]
        if not t1.placed() or not t2.placed():
            continue
        t1_dim, t2_dim = t1.dim(), t2.dim()
        if t1._image.width < t2_dim[0] or t1._image.height < t2_dim[1]:
            lgi('Cannot swap %r with %r due to image size', t1, t2)
            continue
        if t2._image.width < t1_dim[0] or t2._image.height < t1_dim[1]:
            lgi('Cannot swap %r with %r due to image size', t2, t1)
            continue
        lgi('Testing swapping %r and %r', t1, t2)
        score1 = board.score(tile_pos1)
        score2 = board.score(tile_pos2)
        board.swap(tile_pos1, tile_pos2)
        score1_new = board.score(tile_pos1)
        score2_new = board.score(tile_pos2)
        if score1_new + score2_new < score1 + score2:
            lgi('Good swap with tile %s and %s', board.tiles[tile_pos2], board.tiles[tile_pos1])
        else:
            # swap back
            board.swap(tile_pos1, tile_pos2)


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


def bit_count(int_type):
    count = 0
    while int_type:
            int_type &= int_type - 1
            count += 1
    return count


def classify_image(data, offsets=None):
    if offsets:
        col_start, col_end = offsets[0], offsets[1]
        row_start, row_end = offsets[2], offsets[3]
    else:
        col_start, col_end = 0, len(data[0])
        row_start, row_end = 0, len(data)
    min_col = (col_end + col_start) / 2
    min_row = (row_end + row_start) / 2
    colors = [0, 0, 0, 0]
    colors_pos = 0
    total_color_count = 0
    for row_min, row_max in ((row_start, min_row), (min_row, row_end)):
        col_min, col_max = col_start, min_col
        for row in data[row_min:row_max]:
            for val in row[col_min:col_max]:
                colors[colors_pos] += val
        total_color_count += colors[colors_pos]
        colors[colors_pos] /= (row_max-row_min)*(col_max-col_min)
        colors_pos += 1
    for col_min, col_max in ((col_start, min_col), (min_col, col_end)):
        row_min, row_max = row_start, min_row
        for row in data[row_min:row_max]:
            for val in row[col_min:col_max]:
                colors[colors_pos] += val
        total_color_count += colors[colors_pos]
        colors[colors_pos] /= (row_max-row_min)*(col_max-col_min)
        colors_pos += 1

    size = (col_end-col_start)*(row_end-row_start)
    avg = total_color_count / size
    return avg, colors


def compare_ci(img1_ci, img2_ci):
    avg_diff = math.pow(img1_ci[0]-img2_ci[0], 2)
    quad_diff = 0
    for quad in zip(img1_ci[1], img2_ci[1]):
        quad_diff += math.pow(quad[0]-quad[1], 2)
    return math.sqrt(avg_diff+quad_diff)


def classify_large_image(data, width, height):
    info = dict()
    for row in range(0, len(data)-height, height):
        for col in range(0, len(data[0])-width, width):
            ci = classify_image(data, offsets=(col, col+width, row, row+height))
            info[(col, row)] = ci
    return info


class ImageClassifier(object):
    def __init__(self, board, source_image, images, width, height):
        self.board = board
        self.source_image = source_image
        self.width, self.height = width, height
        self.source_image_class = classify_large_image(source_image.data, width, height)
        self.image_classes = list()
        for img in images:
            self.image_classes.append(classify_image(img.data))
        self.image_board_score = defaultdict(dict)
        self.total_score = 0

    def _get_score(self, img_id):
        dim1_col, dim1_row = self.board.coord(img_id)
        if dim1_col == -1:
            return SCORE_UNPLACED
        else:
            if img_id not in self.image_board_score[(dim1_col, dim1_row)]:
                score = compare_ci(self.source_image_class[(dim1_col, dim1_row)], self.image_classes[img_id])
                self.image_board_score[(dim1_col, dim1_row)][img_id] = score
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

    def _get_random_image_ids(self):
        ids = list(range(0, len(self.image_classes)))
        random.shuffle(ids)
        return ids

    def match(self):
        ids = self._get_random_image_ids()
        lgi('Randomly placing images')
        number_rows, number_cols = 0, 0
        for row in range(0, self.source_image.height-self.height, self.height):
            for col in range(0, self.source_image.width-self.width, self.width):
                if number_rows == 0:
                    number_cols += 1
                img_id = ids.pop(0)
                x = self.source_image_class[(col, row)]
                y = self.image_classes[img_id]
                score = compare_ci(x, y)
                self.board.place(img_id, col, row, log=True)
                self.image_board_score[(col, row)][img_id] = score
                self.total_score += score
            number_rows += 1
        lgi('Initial score: %r', self.total_score)
        for t in range(10):
            ids = self._get_random_image_ids()
            cur_score = self.total_score
            while len(ids) >= 2:
                id1, id2 = ids.pop(0), ids.pop(0)
                self._check_swap(id1, id2)
            lgi('Initial score: %r', self.total_score)
            if cur_score == self.total_score:
                break


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


def do_work(source_image, images):
    board = Board(source_image, images)
    min_width, min_height = _get_min_width_and_height(images)
    min_width = 2 * (min_width / 2)
    min_height = 2 * (min_height / 2)
    min_width, min_height = int(min_width / 1.5), int(min_height / 1.5)
    lgi('Min width, height: (%d,%d)', min_width, min_height)
    scaled_images = [board.place(tile.index, -1, -1, min_width, min_height) for tile in board.tiles]
    ic = ImageClassifier(board, source_image, images, min_width, min_height)
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


def compose(image_collection):
    source_image, images = make_images(image_collection)
    lgi('Source image %s', source_image)
    ret = do_work(source_image, images)
    do_check(ret, images)
    return ret

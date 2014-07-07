import sys
from collections import defaultdict, Counter
import math
import random
import datetime
from operator import itemgetter
from models import *
from util import lgi

random.seed(1)

MAX_MILLIS = 8 * 1000 * 1000




def get_slice(data, col_offset, row_offset, width, height):
    ret = list()
    for r in range(height):
        row = list()
        for c in range(width):
            row.append(data[row_offset+r][col_offset+c])
        ret.append(row)
    return ret




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
    start_count = len(free_tiles)
    lgi('Have %d free tiles to fill in slots', start_count)
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
    end_count = len(free_tiles)
    lgi('Filled in spaces with %d tiles.  have %d total remaining free', start_count-end_count, end_count)


def image_histogram_raw(data, offsets=None):
    c = [0 for _ in range(256)]
    if offsets:
        row_start, col_start, row_end, col_end = offsets
    else:
        row_start, row_end = 0, len(data)
        col_start, col_end = 0, len(data[0])
    for row in data[row_start:row_end]:
        for val in row[col_start:col_end]:
            c[val] += 1
    return c, (row_end-row_start)*(col_end-col_start)


def image_histogram(data, offsets=None):
    c, img_size = image_histogram_raw(data, offsets)
    side_buffers = list()
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


def find_medians(data, offsets=None, dim=(3, 3)):
    if offsets:
        row_start, col_start, row_end, col_end = offsets
    else:
        row_start, row_end = 0, len(data)
        col_start, col_end = 0, len(data[0])
    delta_col = 1.0 * (col_end-col_start) / dim[0]
    delta_row = 1.0 * (row_end-row_start) / dim[1]
    lgi('Delta row: %r, delta col: %r for size (%d,%d)', delta_row, delta_col, col_end-col_start, row_end-row_start)
    ret = list()
    top_row = 0
    for top_row_pos in range(dim[1]):
        top_col = 0
        space_vals = list()
        for top_col_pos in range(dim[0]):
            # now count inside box
            t = 0
            c = 0
            row = top_row
            while row < (top_row+delta_row):
                col = top_col
                row_int = int(row)
                if row_int >= len(data):
                    lgi('Breaking with row %d', row_int)
                    break
                while col < (top_col+delta_col):
                    col_int = int(col)
                    if col_int >= len(data[0]):
                        lgi('Breaking with col %d', col_int)
                        break
                    t += data[row_int][col_int]
                    c += 1
                    col += 1
                row += 1
                need_number_cols = False
            if c == 0:
                lgi('no data for row %r col %r', row, col)
                break
            avg = 1.0*t/c
            space_vals.append(int(avg))
            top_col += delta_col
        ret.append(space_vals)
        top_row += delta_row
    return ret


def quadrant_histogram(data, offsets=None):
    """ 0: upper left, 1: upper right, 2: lower left, 3: lower right
    """
    if offsets:
        row_start, col_start, row_end, col_end = offsets
    else:
        row_start, row_end = 0, len(data)
        col_start, col_end = 0, len(data[0])
    mid_row, mid_col = (row_end+row_start)/2, (col_end+col_start)/2
    quads = list()
    q = [0 for _ in range(256)]
    for row in data[row_start:mid_row]:
        for val in row[col_start:mid_col]:
            q[val] += 1
    quads.append(q)
    q = [0 for _ in range(256)]
    for row in data[row_start:mid_row]:
        for val in row[mid_col:col_end]:
            q[val] += 1
    quads.append(q)
    q = [0 for _ in range(256)]
    for row in data[mid_row:row_end]:
        for val in row[col_start:mid_col]:
            q[val] += 1
    quads.append(q)
    q = [0 for _ in range(256)]
    for row in data[mid_row:row_end]:
        for val in row[mid_col:col_end]:
            q[val] += 1
    quads.append(q)
    # now do middle 1/3rd
    middle_height, middle_width = (row_end-row_start)/3, (col_end-col_start)/3
    r_0, r_1 = row_start+middle_height, row_end-middle_height
    c_0, c_1 = col_start+middle_width, col_end-middle_width
    q = [0 for _ in range(256)]
    for row in data[r_0:r_1]:
        for val in row[c_0:c_1]:
            q[val] += 1
    quads.append(q)
    #lgi('Sizes for quads for (%d,%d,%d,%d) : %s', row_start, col_start, row_end, col_end, str([len(_) for _ in quads]))
    return quads


def find_median(histogram, number_elements=None):
    if number_elements is None:
        number_elements = 0
        for val in histogram:
            number_elements += val
    cume = 0
    for color, count in enumerate(histogram):
        cume += count
        if cume >= number_elements/2:
            return color
    # should not happen
    return 255


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
    def __init__(self, board, timer):
        self.board = board
        self.image_board_score = defaultdict(dict)
        self.total_score = 0
        self.number_images = len(board.tiles)
        self.total_score = 0
        self.timer = timer
        self.cached_board_scores = dict()
        self.cached_image_scores = dict()
        self.cached_image_overlay_scores = [dict() for _ in range(self.number_images)]

    def _get_img_score(self, img_id):
        key = img_id
        if key in self.cached_image_scores:
            return self.cached_image_scores[key]
        medians = list()
        for quad in quadrant_histogram(self.board.tiles[img_id]._image):
            medians.append(find_median(quad))
        self.cached_image_scores[key] = medians
        return medians

    def _get_board_score(self, row, col, width, height):
        key = row, col, width, height
        if key in self.cached_board_scores:
            return self.cached_board_scores[key]
        medians = list()
        for quad in quadrant_histogram(self.board.source_image, offsets=(row, col, row+height, col+width)):
            medians.append(find_median(quad))
        self.cached_board_scores[key] = medians
        return medians

    def _get_score(self, img_id, width, height):
        row, col = self.board.coord(img_id)
        if row == -1:
            return SCORE_UNPLACED
        d = self.cached_image_overlay_scores[img_id]
        key = row, col, width, height
        if key in d:
            return d[key]
        scorings_diff = 0
        for a, b in zip(self._get_board_score(key[0], key[1], width, height), self._get_img_score(img_id)):
            scorings_diff += (a-b)**2
        d[key] = scorings_diff
        return scorings_diff

    def _check_swap(self, img1, img2, width, height):
        score1_pre, score2_pre = self._get_score(img1, width, height), self._get_score(img2, width, height)
        img1_coord, img2_coord = self.board.coord(img1), self.board.coord(img2)
        self.board.swap(img1, img2)
        score1_post, score2_post = self._get_score(img1, width, height), self._get_score(img2, width, height)
        delta_score = (score1_pre + score2_pre) - (score1_post + score2_post)
        lgi('Swapped ids %r and %r, pre: (%r,%r), post: (%r,%r) => %r', img1, img2, score1_pre, score2_pre, score1_post, score2_post, delta_score)
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

    def _get_random_placed_coords(self, width, height):
        placed_coords = list()
        for row in range(0, len(self.board.source_image)-height, height):
            for col in range(0, len(self.board.source_image[0])-width, width):
                placed_coords.append((row, col))
        random.shuffle(placed_coords)
        return placed_coords

    def match(self, width, height):
        for t in range(10):
            number_swaps = 0
            ids = _get_random_image_ids(self.number_images)
            while ids:
                id1 = ids.pop(0)
                for row, col in self._get_random_placed_coords(width, height):
                    if self.timer.should_finish():
                        break
                    tile = self.board.get_tile_at(row, col)
                    if tile is None:
                        # should check if this bumps into another piece
                        self.board.place(id1, row, col, width, height)
                        # just get them all down first -- watch out as might not do that for all positions
                        number_swaps += 1
                        break
                    else:
                        b2 = tile.dim()
                        if b2[0] == width and b2[1] == height:
                            # same size, now just check if can swap
                            if self._check_swap(id1, tile.index, width, height):
                                number_swaps += 1
                            continue
                        # difference in size, for now don't allow.  need to find three other tiles to help out
                if self.timer.should_finish():
                    break
            lgi('Score in round %r after %d swaps: %r', t, number_swaps, self.total_score)
            if self.timer.should_finish():
                lgi('Breaking due to time')
                break
            if number_swaps <= 1:
                break
            if t >= 20:
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
    lgi('Source image: %r', source_image)
    board = Board(source_image, images)
    TiledBoard(board, 21, 11)
    sys.exit(1)
    min_width, min_height = _get_min_width_and_height(images)
    min_width = 2 * (min_width / 2)
    min_height = 2 * (min_height / 2)
    min_width, min_height = int(min_width/2), int(min_height/2)
    lgi('Min width, height: (%d,%d)', min_width, min_height)
    #scaled_images = [board.place(tile.index, -1, -1, min_width, min_height) for tile in board.tiles]
    ic = ImageClassifier(board, timer)
    #ic.match()
    #ic.match2(overall_histo, per_image_histo, min_width, min_height)
    ic.match(min_width, min_height)
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
        timer = MyTimer(MAX_MILLIS)
        source_image, images = make_images(image_collection)
        lgi('Source image %s', source_image)
        ret = do_work(source_image, images, timer)
        lgi('Total time: %r', timer.delta()/1000)
        #do_check(ret, images)
        return ret

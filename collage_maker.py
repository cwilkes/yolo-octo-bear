import sys
from collections import defaultdict, Counter
import math


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
            raise Exception('Asking for tile_pos %d in array size %d' % (tile_pos, len(self.tiles)))
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
        for col_min, col_max in ((col_start, min_col), (min_col, col_end)):
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
    info = defaultdict(dict)
    for row in range(0, len(data)-height, height):
        for col in range(0, len(data[0])-width, width):
            ci = classify_image(data, offsets=(col, col+width, row, row+height))
            info[row][col] = ci
    return info


def match_by_ci(board, lci, small_cis):
    matches = dict()
    free_tiles = dict()
    for tile in board.tiles:
        free_tiles[tile.index] = tile
    ending_col, ending_row = 0, 0
    by_img = dict()
    for img_index, img_ci in small_cis.items():
        best_key, best_score = None, sys.maxint
        for row in lci.keys():
            for col, ci in lci[row].items():
                key = col, row
                if key in matches:
                    continue
                comp = compare_ci(ci, img_ci)
                if comp < best_score:
                    best_key, best_score = key, comp
        if best_key is None:
            break
        matches[best_key] = img_index, best_score
        by_img[img_index] = best_key[0], best_key[1], best_score
        free_tiles.pop(img_index)
        img = board.place(img_index, best_key[0], best_key[1], log=True)
        ending_col = max(ending_col, best_key[0]+img.width)
        ending_row = max(ending_row, best_key[1] + img.height)
    # for now just go back through the end to the beginning
    for img_index, img_ci in reversed(small_cis.items()):
        try:
            cur_col, cur_row, cur_score = by_img[img_index]
        except:
            cur_col, cur_row, cur_score = -1, -1, 0
        for row in lci.keys():
            for col, ci in lci[row].items():
                key = col, row
                existing = matches[key]
                new_score_a = compare_ci(ci, img_ci)
                new_score_b = compare_ci(ci, )

                if comp < best_score:
                    best_key, best_score = key, comp
        if best_key is None:
            break
        matches[best_key] = img_index
        free_tiles.pop(img_index)
        img = board.place(img_index, best_key[0], best_key[1], log=True)
    return free_tiles, ending_col, ending_row


def do_work(source_image, images):
    board = Board(source_image, images)
    min_width, min_height = _get_min_width_and_height(images)
    min_width = 2 * (min_width / 2)
    min_height = 2 * (min_height / 2)
    min_width, min_height = int(min_width / 1.5), int(min_height / 1.5)
    lgi('Min width, height: (%d,%d)', min_width, min_height)
    scaled_images = [board.place(tile.index, -1, -1, min_width, min_height) for tile in board.tiles]
    img_classifications = dict()
    for img in scaled_images:
        ci = classify_image(img.data)
        lgi('small class, %r : %r', img, ci)
        img_classifications[img.index] = ci
    lci = classify_large_image(source_image.data, min_width, min_height)
    free_tiles, ending_col, ending_row = match_by_ci(board, lci, img_classifications)
    fill_sides(board, free_tiles.values(), ending_col, ending_row)
    return board.as_ary()


def do_work_old2(source_image, images):
    board = Board(source_image, images)
    num_nodes = 7
    width = source_image.width / num_nodes + 1
    height = source_image.height / num_nodes + 1
    scaled_images = list()
    for tile_pos in range(len(board.tiles)):
        scaled_images.append(board.place(tile_pos, -1, -1, width, height))
    thresholds = [255/3, 2*255/3]
    board_scores = board.threshold_scoring(width, height, thresholds)
    scaled_images_scores = [img.threshold_scoring(thresholds) for img in scaled_images]
    comp = list()
    for img_scores in scaled_images_scores:
        c2 = list()
        for row, col, board_score2 in board_scores:
            score = 0
            for a, b in zip(img_scores, board_score2):
                score += (a-b)*(a-b)
            c2.append(score)
        comp.append(c2)
    used_images, used_board = set(), set()
    ending_col, ending_row = 0, 0
    while len(used_images) < len(board_scores):
        min_img_pos, min_board_pos, min_score = None, None, sys.maxint
        for img_pos, c2 in enumerate(comp):
            if img_pos in used_images:
                continue
            for board_pos, val in enumerate(c2):
                if board_pos in used_board:
                    continue
                if val < min_score:
                    min_img_pos, min_board_pos, min_score = img_pos, board_pos, val
        if min_img_pos is None:
            break
        c, r, score = board_scores[min_board_pos]
        ending_col = max(ending_col, c)
        ending_row = max(ending_row, r)
        board.place(min_img_pos, c, r, log=True)
        used_images.add(min_img_pos)
        used_board.add(min_board_pos)
    lgi('Used images: (%d): %s', len(used_images), sorted(used_images))
    lgi('Used board: (%d): %s', len(used_board), sorted(used_board))
    # fill in sides
    free_tiles = list()
    for tile in board.tiles:
        if not tile.placed():
            free_tiles.append(tile)
    row = 0
    ending_col += width
    ending_row += height
    lgi('Filling in right col %r', ending_col)
    while row < source_image.height:
        tile = free_tiles.pop(0)
        img = tile._image
        h = min(img.height, source_image.height-row)
        board.place(tile._image.index-1, ending_col, row, source_image.width-ending_col, h, True)
        row += h
    col = 0
    lgi('Filling in bottom row %r', ending_row)
    while col < ending_col:
        tile = free_tiles.pop(0)
        img = tile._image
        w = min(img.width, ending_col-col)
        board.place(tile._image.index-1, col, ending_row, w, source_image.height-ending_row, True)
        col += w
    simulated_annealing(board, 7)
    simulated_annealing(board, -7)
    simulated_annealing(board, 6)
    simulated_annealing(board, -6)
    simulated_annealing(board, 5)
    simulated_annealing(board, -5)
    return board.as_ary()


def do_work_old(source_image, images):
    board = Board(source_image, images)
    num_nodes = 14
    width = source_image.width / num_nodes + 1
    height = source_image.height / num_nodes + 1
    row = 0
    col = 0
    index = 0
    lgi('Width: %d, height: %d', width, height)
    ending_col, ending_row = None, -1
    while row < source_image.height and index < len(images):
        img = board.tiles[index]._image
        if img.width < width or img.height < height:
            lgi('Not placing image %s as smaller than (%d,%d)', img, width, height)
            index += 1
            continue
        w = width
        if col + w > source_image.width:
            ending_col = col
            index += 1
            col = 0
            row += h
            continue
        h = height
        if row + h > source_image.height:
            break
        board.place(index, col, row, w, h)
        ending_row = max(ending_row, row)
        col += w
        index += 1
    lgi('Now filling in with index %d ending_col %r, ending_row %r', index, ending_col, ending_row)
    row = 0
    while row < source_image.height:
        img = board.tiles[index]._image
        h = min(img.height, source_image.height-row)
        board.place(index, ending_col, row, source_image.width-ending_col, h, True)
        row += h
        index += 1
    col = 0
    ending_row += height
    while col < ending_col:
        img = board.tiles[index]._image
        w = min(img.width, ending_col-col)
        board.place(index, col, ending_row, w, source_image.height-ending_row, True)
        col += w
        index += 1
    lgi('Done with initial placement')
    # now rescale all their image sizes
    if False:
        simulated_annealing(board, 8)
        simulated_annealing(board, -8)
        simulated_annealing(board, 4)
        simulated_annealing(board, -4)
        simulated_annealing(board, 2)
        simulated_annealing(board, -2)
    for score in board.threshold_scoring(22, 17, [85, 170]):
        lgi('Score: %s', score)

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

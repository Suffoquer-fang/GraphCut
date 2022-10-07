import numpy as np
import networkx as nx 
from PIL import Image
import random
from scipy.signal import *


INF = 1e8
EPS = 1e-8


class Point: 
    def __init__(self, x, y):
        self.x = x 
        self.y = y
        self.idx = (x, y)
    

    def neighbors(self):
        i = self.x 
        j = self.y
        return [Point(i-1, j), Point(i+1, j), Point(i, j-1), Point(i, j+1)]

    def left_nbr(self):
        return Point(self.x, self.y-1)

    def top_nbr(self):
        return Point(self.x-1, self.y)



class Region:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1 
        self.x2 = x2 
        self.y1 = y1 
        self.y2 = y2 

    def contains(self, x, y):
        return self.x1 <= x and x < self.x2 and self.y1 <= y and y < self.y2 
    
    def slice(self):
        return (slice(self.x1, self.x2), slice(self.y1, self.y2))

    def points_iter(self):
        for x in range(self.x1, self.x2):
            for y in range(self.y1, self.y2):
                yield Point(x, y)

    def clip(self, x1, y1, x2, y2):
        self.x1 = max(self.x1, x1)
        self.y1 = max(self.y1, y1)
        self.x2 = min(self.x2, x2)
        self.y2 = min(self.y2, y2)

class SeamMap:
    def __init__(self, height, width):
        self.has_left = np.full([height, width], False, np.bool)
        self.has_top = np.full([height, width], False, np.bool)
        self.left_nbr_cost = np.full([height, width], 0, np.float32)
        self.top_nbr_cost = np.full([height, width], 0, np.float32)
        self.offset = np.full([height, width, 2], 0)
    


def show(a):
    print(a.astype(np.uint8))


def get_grad(im):
    # im = im.convert('L')
    im = im.astype(np.float)
    grad_x = np.gradient(im, axis=0)
    grad_y = np.gradient(im, axis=1)
    grad_x = np.sum(grad_x ** 2, axis=2)
    grad_y = np.sum(grad_y ** 2, axis=2)

    # grad_x = np.sqrt(grad_x)
    # grad_y = np.sqrt(grad_y)

    return grad_x, grad_y


def get_l2_energy_cost(s: Point, t: Point, im_src: np.ndarray, im_dst: np.ndarray, A_grad = None, B_grad = None, use_grad = False) -> np.float:
    # print(s.idx, t.idx)
    cost = np.square(im_src[s.idx] - im_dst[s.idx]).sum(-1) + np.square(im_src[t.idx] - im_dst[t.idx]).sum(-1)
    
    if use_grad:
        # A_grad = get_grad(im_src)
        # B_grad = get_grad(im_dst)
        cost = get_grad_cost(s.x, s.y, t.x, t.y, cost, A_grad, B_grad)
    
    return cost

def get_l2_norm_cost(x_s, y_s, x_t, y_t, im_diff):
    # M = 0
    return im_diff[x_s, y_s] + im_diff[x_t, y_t]
    
    


def get_grad_cost(x_s, y_s, x_t, y_t, M, A_grad, B_grad):
    
    A_grad_x, A_grad_y = A_grad
    B_grad_x, B_grad_y = B_grad

    
    if x_s == x_t:
        grad_src = A_grad_y
        grad_dst = B_grad_y
    else:
        grad_src = A_grad_x
        grad_dst = B_grad_x 

    grad_sum = grad_src[x_s, y_s] + grad_src[x_t, y_t] + grad_dst[x_s, y_s] + grad_dst[x_t, y_t]

    return M / (grad_sum + EPS)
    # return M


def get_bound_box(im_map) -> Region:
    if not im_map.any(): return None
    temp_coords = np.where(im_map > 0)
    r, c = temp_coords
    x_min, x_max = min(r), max(r)
    y_min, y_max = min(c), max(c)
    return Region(x_min, y_min, x_max + 1, y_max + 1)

def handle_surrouned_region(src_map, dst_map):
    x_min, y_min, x_max, y_max = get_bound_box(dst_map)
    x_center, y_center = (x_min + x_max) // 2, (y_min + y_max) // 2
    assert(src_map[x_center, y_center] == True)

    src_map[x_center, y_center] = False


def get_seam_node(u: Point, v: Point, im_src: np.ndarray) -> Point:
    height, width, _ = im_src.shape
    
    return Point(u.x * height + v.x, u.y * width + v.y)

def build_graph(im_src, src_map, im_input, offset, seam_map: SeamMap, use_old_cut = True, use_grad = False):
    height, width = im_src[:, :, 0].shape
    im_dst, dst_map = handle_input_offset(height, width, im_input, offset)

    overlap_map = src_map & dst_map
    if (overlap_map == src_map).all() or (not overlap_map.any()):
        im_src[:, :] = im_dst[:, :]
        # show(src_map)
        # show(dst_map)
        # show(overlap_map)
        src_map |= dst_map
        # print('ww')
        return None 
    if (overlap_map == dst_map).all(): 
        # surrounded region
        print('Handle Surrouned')
        # handle_surrouned_region(src_map, dst_map)


    # calculate grad


    A_grad = get_grad(im_src)
    B_grad = get_grad(im_dst)

    

    overlap_region = get_bound_box(overlap_map)

    
    map_region = Region(0, 0, height, width)

    im_diff = (im_src - im_dst) ** 2
    im_diff = np.sum(im_diff, axis = 2)

    
    super_src = Point(-1, -1)
    super_dst = Point(height, width)

    G = nx.Graph()

    for curr in overlap_region.points_iter():
        # curr = Point(-1, -1)
        if not overlap_map[curr.idx]: continue
        left_nbr = curr.left_nbr()
        top_nbr = curr.top_nbr()

        connect_src = False
        connect_dst = False

        for nbr in curr.neighbors():
            if not map_region.contains(nbr.x, nbr.y): continue 
            
            if not overlap_map[nbr.idx] and src_map[nbr.idx]: connect_src = True
            if not overlap_map[nbr.idx] and dst_map[nbr.idx]: 
                # print(nbr.idx)
                connect_dst = True

        if connect_src and not connect_dst:
            G.add_edge(super_src.idx, curr.idx, weight=INF) 
        if connect_dst and not connect_src:
            G.add_edge(super_dst.idx, curr.idx, weight=INF)

        
        if map_region.contains(left_nbr.x, left_nbr.y):
            if overlap_map[left_nbr.idx]:
                if use_old_cut and seam_map.has_left[curr.idx]:

                    seam_node = get_seam_node(left_nbr, curr, im_src)
                    
                    left_offset = seam_map.offset[left_nbr.idx]
                    curr_offset = seam_map.offset[curr.idx]

                    left_patch, _ = handle_input_offset(height, width, im_input, left_offset)
                    curr_patch, _ = handle_input_offset(height, width, im_input, curr_offset)

                    seam_to_dst_cost = get_l2_energy_cost(left_nbr, curr, left_patch, curr_patch)
                    left_to_seam_cost = get_l2_energy_cost(left_nbr, curr, left_patch, im_dst)
                    seam_to_curr_cost = get_l2_energy_cost(left_nbr, curr, im_dst, curr_patch)


                    G.add_edge(seam_node.idx, super_dst.idx, weight=seam_to_dst_cost)
                    G.add_edge(left_nbr.idx, seam_node.idx, weight=left_to_seam_cost)
                    G.add_edge(seam_node.idx, curr.idx, weight=seam_to_curr_cost)



                    # pass 
                else: 
                    energe_cost = get_l2_energy_cost(left_nbr, curr, im_src, im_dst, A_grad, B_grad, use_grad)
                    G.add_edge(left_nbr.idx, curr.idx, weight = energe_cost)


           

        if map_region.contains(top_nbr.x, top_nbr.y):
            if overlap_map[top_nbr.idx]:
                if use_old_cut and seam_map.has_top[curr.idx]:
                    seam_node = get_seam_node(top_nbr, curr, im_src)
                    
                    top_offset = seam_map.offset[top_nbr.idx]
                    curr_offset = seam_map.offset[curr.idx]

                    top_patch, _ = handle_input_offset(height, width, im_input, top_offset)
                    curr_patch, _ = handle_input_offset(height, width, im_input, curr_offset)

                    seam_to_dst_cost = get_l2_energy_cost(top_nbr, curr, top_patch, curr_patch)
                    top_to_seam_cost = get_l2_energy_cost(top_nbr, curr, top_patch, im_dst)
                    seam_to_curr_cost = get_l2_energy_cost(top_nbr, curr, im_dst, curr_patch)


                    G.add_edge(seam_node.idx, super_dst.idx, weight=seam_to_dst_cost)
                    G.add_edge(top_nbr.idx, seam_node.idx, weight=top_to_seam_cost)
                    G.add_edge(seam_node.idx, curr.idx, weight=seam_to_curr_cost)
                else: 
                    energe_cost = get_l2_energy_cost(top_nbr, curr, im_src, im_dst, A_grad, B_grad, use_grad)
                    G.add_edge(top_nbr.idx, curr.idx, weight = energe_cost)
           
    src_map |= dst_map
    
    return G



def update_seam_map(G:nx.Graph, seam_map:SeamMap, src_set: set, dst_set: set, curr:Point, im_src, left=True):
    left_nbr = curr.left_nbr()
    top_nbr = curr.top_nbr()
    if curr.idx in dst_set: 
        if left:
            if left_nbr.idx in src_set:
                seam_map.has_left[curr.idx] = True 
                seam_node = get_seam_node(left_nbr, curr, im_src)
                if G.has_node(seam_node.idx):
                    temp_node = curr if seam_node in src_set else left_nbr 
                    seam_map.left_nbr_cost[curr.idx] = G[seam_node.idx][temp_node.idx]['weight']
                else:
                    seam_map.left_nbr_cost[curr.idx] = G[left_nbr.idx][curr.idx]['weight']
            else: 
                seam_map.has_left[curr.idx] = False 
                seam_map.left_nbr_cost[curr.idx] = 0
        else: 
            if top_nbr.idx in src_set:
                seam_map.has_top[curr.idx] = True 
                seam_node = get_seam_node(top_nbr, curr, im_src)
                if G.has_node(seam_node.idx):
                    temp_node = curr if seam_node in src_set else top_nbr 
                    seam_map.top_nbr_cost[curr.idx] = G[seam_node.idx][temp_node.idx]['weight']
                else:
                    seam_map.top_nbr_cost[curr.idx] = G[top_nbr.idx][curr.idx]['weight']
            else:
                seam_map.has_top[curr.idx] = False
                seam_map.top_nbr_cost[curr.idx] = 0
    

    
def patch_fitting(im_src, src_map, im_input, offset, seam_map: SeamMap, use_old_cut = True, use_grad = False):
    # overlap_map = src_map & dst_map
    if use_old_cut: use_grad = False 
    G = build_graph(im_src, src_map, im_input, offset, seam_map)
    

    height, width = im_src[:, :, 0].shape
    h, w, _ = im_input.shape

    if G:
        super_src = Point(-1, -1)
        super_dst = Point(height, width)
        if not (G.has_node(super_src.idx) and G.has_node(super_dst.idx)):
            print('fitting nothing....') 
            return 
        
        _, partion = nx.minimum_cut(G, super_src.idx, super_dst.idx, 'weight')
        left, right = partion


    off_x, off_y = offset
    copy_region = Region(off_x, off_y, off_x + h, off_y + w)
    copy_region.clip(0, 0, height, width)

    for curr in copy_region.points_iter():
        
        if not G or not G.has_node(curr.idx):
            im_src[curr.idx] = im_input[curr.x - off_x, curr.y - off_y]
            seam_map.offset[curr.idx] = offset
            seam_map.has_left[curr.idx] = False
            seam_map.has_top[curr.idx] = False
            seam_map.top_nbr_cost[curr.idx] = 0
            seam_map.left_nbr_cost[curr.idx] = 0
        else: 
            if curr.idx in right:
                im_src[curr.idx] = im_input[curr.x - off_x, curr.y - off_y]
                seam_map.offset[curr.idx] = offset

                update_seam_map(G, seam_map, left, right, curr, im_src, True)
                update_seam_map(G, seam_map, left, right, curr, im_src, False)

                right_nbr = Point(curr.x, curr.y + 1)
                bottom_nbr = Point(curr.x + 1, curr.y)
                if right_nbr.idx in left:
                    update_seam_map(G, seam_map, right, left, right_nbr, im_src, True)    
                if bottom_nbr.idx in left:
                    update_seam_map(G, seam_map, right, left, bottom_nbr, im_src, False)

    return im_src

def handle_input_offset(height, width, im_input, offset):
    dst_map = np.zeros([height, width]).astype(np.bool)
    im_dst = np.zeros([height, width, 3])

    h, w, _ = im_input.shape
    off_x, off_y = offset

    
    dst_map[max(0, off_x):min(off_x+h, height), max(0, off_y):min(off_y+w, width)] = 1
    im_dst[max(0, off_x):min(off_x+h, height), max(0, off_y):min(off_y+w, width)] = im_input[max(0, -off_x):min(h, height - off_x), max(0, -off_y):min(w, width - off_y)]

    return im_dst, dst_map
    
def update_src_map(src_map, dst_map):
    src_map |= dst_map

def get_offset_random(im_src, src_map, im_input):
    if not src_map.any(): return (0, 0)
    h, w, _ = im_input.shape
    temp = np.where(src_map == 0)
    r, c = temp

    off_x_max = h // 2
    off_y_max = w // 2
    offset = [random.randint(r[0]-h, r[0]-1), random.randint(c[0]-w, c[0]-1)]
    return offset

def get_cost(im_src, im_dst, src_map, dst_map):
    overlap_map = src_map & dst_map
    if not overlap_map.any(): return INF 
    im_diff = (im_src - im_dst) ** 2
    im_diff = np.sum(im_diff, axis = 2)
    # im_diff = np.sqrt(im_diff)
    At = im_diff[overlap_map]

    return np.mean(At)


def get_conv(im_src, src_map, im_input):
    rev_im_input = im_input[::-1, ::-1]
    h, w, _ = im_input.shape
    rev_dst_map = np.ones([h, w])


    im_src_square = im_src.astype(np.float) ** 2
    im_input_square = im_input.astype(np.float) ** 2


    rev_im_input_square = im_input_square[::-1, ::-1]

    conv_overlap = fftconvolve(src_map, rev_dst_map)
    h, w = conv_overlap.shape

    conv_src = np.zeros([h, w, 3])
    conv_dst = np.zeros([h, w, 3])
    conv_cross = np.zeros([h, w, 3])

    for c in range(3):

        conv_src[:,:,c] = fftconvolve(im_src_square[:,:,c], rev_dst_map)
        conv_dst[:,:,c] = fftconvolve(src_map, rev_im_input_square[:,:,c])
        conv_cross[:,:,c] = fftconvolve(im_src[:,:,c], rev_im_input[:,:,c])

    return conv_src, conv_dst, conv_cross, conv_overlap

def get_cost_fft_based(conv_src, conv_dst, conv_cross, conv_overlap, offset, im_input):
    off_x, off_y = offset
    h, w, _ = im_input.shape 
    x, y = off_x + h - 1, off_y + w - 1
    At = conv_src[x, y] + conv_dst[x, y] - 2 * conv_cross[x, y]
    At = np.sum(At, axis=-1)
    # print(conv_src[x, y], conv_dst[x, y], conv_cross[x, y])
    
    card = conv_overlap[x, y]
    # print(card)
    # assert(card > 0)
    # assert(At > 0)
    # if card <= 0: return EPS
    # assert(False)
    return At / card
    
    
def get_offset_entire_matching(im_src, src_map, im_input, error_region: Region):
    height, width, _ = im_src.shape
    h, w, _ = im_input.shape
    if not src_map.any(): return (0, 0)
    
    if not error_region:
        temp = np.where(src_map == 0)
        r, c = temp
        off_x_min = int(r[0] - 0.75 * h)
        off_x_max = int(r[0] - 0.25 * h)
        off_y_min = int(c[0] - 0.75 * w)
        off_y_max = int(c[0] - 0.25 * w)
    else:
        x1, y1, x2, y2 = error_region.x1, error_region.y1, error_region.x2, error_region.y2
        off_x_min = int(x2 - h)
        off_x_max = int(x1)
        off_y_min = int(y2 - w)
        off_y_max = int(y1)
    

    sigma = np.var(im_input)
    k = 0.01

    prob_map = np.zeros([h + height, w + width])
    idx_map = list(range(0, (h + height) * (w + width)))


    conv_src, conv_dst, conv_cross, conv_overlap = get_conv(im_src, src_map, im_input)

    for x in range(off_x_min, off_x_max):
        for y in range(off_y_min, off_y_max):
            # print(x, y)
            im_dst, dst_map = handle_input_offset(height, width, im_input, (x, y))
            # cost1 = get_cost(im_src, im_dst, src_map, dst_map)
            offset = (x, y)
            cost2 = get_cost_fft_based(conv_src, conv_dst, conv_cross, conv_overlap, offset, im_input)

            # print(cost1, cost2)
            prob = np.exp(-cost2 / (k * sigma + EPS))

            off_x = x - off_x_min
            off_y = y - off_y_min

            prob_map[off_x, off_y] = prob 

    prob_map /= np.sum(prob_map)
    # assert(np.sum(prob_map) == 1)
    # print(np.sum(prob_map))

    idx = np.random.choice(idx_map, size=1, p=prob_map.reshape(-1))
    # idx = np.argmax(prob_map.reshape(-1))
    off_x, off_y = np.unravel_index(idx, prob_map.shape)
    x, y = off_x + off_x_min, off_y + off_y_min
    return (int(x), int(y))


def get_error_cost(seam_map:SeamMap, error_region: Region):
    ret = 0
    for curr in error_region.points_iter():
        # curr = Point(-1, -1)
        left_nbr = curr.left_nbr()
        top_nbr = curr.top_nbr()
        if seam_map.has_left[curr.idx] and error_region.contains(left_nbr.x, left_nbr.y): ret += seam_map.left_nbr_cost[curr.idx]
        if seam_map.has_top[curr.idx] and error_region.contains(top_nbr.x, top_nbr.y): ret += seam_map.top_nbr_cost[curr.idx]

    return ret

def get_error_cost_fft_based(conv_left, conv_top, offset, region_size):
    rh, rw = region_size 
    
    off_x, off_y = offset
    x, y = off_x + rh - 1, off_y + rw - 1
    return conv_left[x, y] + conv_top[x, y]
    


def get_first_unconverd_pixel(src_map:np.ndarray) -> Point:
    if src_map.all(): return Point(-1, -1)
    temp = np.where(src_map == 0)
    r, c = temp
    return Point(r[0], c[0])

def get_conv_error_region(seam_map: SeamMap, region_size):
    rh, rw = region_size 
    region_map = np.ones([rh, rw]).astype(np.float32)
    conv_left = fftconvolve(seam_map.left_nbr_cost, region_map)
    conv_top = fftconvolve(seam_map.top_nbr_cost, region_map)

    return conv_left, conv_top
    

def get_error_region(im_src, src_map: np.ndarray, seam_map:SeamMap, region_size) -> Region:
    rh, rw = region_size 
    height, width = src_map.shape

    conv_left, conv_top = get_conv_error_region(seam_map, region_size)

    ret_x = 0
    ret_y = 0
    # print('searching error region')
    temp = get_first_unconverd_pixel(src_map)
    if temp.idx == (-1, -1):
        max_error = -1
        for x in range(temp.x + 1, height - rh):
            for y in range(temp.y + 1, width - rw):
                # temp_error = get_error_cost(seam_map, Region(x, y, x + rh, y + rw))
                
                temp_error = get_error_cost_fft_based(conv_left, conv_top, (x, y), region_size)
                if temp_error > max_error:
                    max_error = temp_error
                    ret_x, ret_y = x, y 
        ret_x += rh // 2
        ret_y += rw // 2
    else: 
        ret_x = temp.x - rh // 2
        ret_y = temp.y - rw // 2

    ret_x = max(0, ret_x)
    ret_y = max(0, ret_y)
    ret_x = min(height - rh, ret_x)
    ret_y = min(width - rw, ret_y)
    return Region(ret_x, ret_y, ret_x + rh, ret_y + rw)

    
def get_offset_subpatch_matching(im_src, src_map, im_input, error_region: Region, region_size):
    if not src_map.any(): return (0, 0)

    h, w, _ = im_input.shape
    rh, rw = region_size

    im_src_subpatch = np.zeros(im_src.shape)
    dst_map = np.ones([h, w]).astype(np.bool)

    region_slice = error_region.slice()
    im_src_subpatch = im_src[region_slice]

    

    conv_src, conv_dst, conv_cross, conv_overlap = get_conv(im_input, dst_map, im_src_subpatch)
    off_x_min = 0
    off_x_max = h - rh 
    off_y_min = 0
    off_y_max = w - rw

    sigma = np.var(im_src_subpatch)
    k = 0.01

    prob_map = np.zeros([h + rh, w + rw])
    idx_map = list(range(0, (h + rh) * (w + rw)))

    for x in range(off_x_min, off_x_max):
        for y in range(off_y_min, off_y_max):
            
            # cost1 = get_cost(im_src, im_dst, src_map, dst_map)
            offset = (x, y)
            cost2 = get_cost_fft_based(conv_src, conv_dst, conv_cross, conv_overlap, offset, im_src_subpatch)

            # print(cost1, cost2)
            prob = np.exp(-cost2 / (k * sigma + EPS))

            off_x = x - off_x_min
            off_y = y - off_y_min

            prob_map[off_x, off_y] = prob 

    prob_map /= np.sum(prob_map)
    # assert(np.sum(prob_map) == 1)
    # print(np.sum(prob_map))

    idx = np.random.choice(idx_map, size=1, p=prob_map.reshape(-1))
    # idx = np.argmax(prob_map.reshape(-1))
    off_x, off_y = np.unravel_index(idx, prob_map.shape)
    x, y = off_x + off_x_min, off_y + off_y_min


    true_x, true_y = error_region.x1 - x, error_region.y1 - y
    return (int(true_x), int(true_y))



def debug_error_region(im_src, error_region:Region):
    im_temp = im_src.copy()
    x1, y1, x2, y2 = error_region.x1, error_region.y1, error_region.x2, error_region.y2
    im_temp[x1-1:x1+1, y1:y2] = (255, 0, 0)
    im_temp[x2-1:x2+1, y1:y2] = (255, 0, 0)
    im_temp[x1:x2, y1-1:y1+1] = (255, 0, 0)
    im_temp[x1:x2, y2-1:y2+1] = (255, 0, 0)
    Image.fromarray(im_temp.astype(np.uint8)).show()
    return Image.fromarray(im_temp.astype(np.uint8))

def debug_cut(im_src, seam_map:SeamMap):
    im_temp = im_src.copy() 
    im_temp[seam_map.has_left] = (0, 255, 0)
    im_temp[seam_map.has_top] = (0, 255, 0)
    Image.fromarray(im_temp.astype(np.uint8)).show()
    return Image.fromarray(im_temp.astype(np.uint8))



def test_surrounded():
    # im = Image.open('data/strawberries2.jpg')
    im = Image.open('data/akeyboard_small.jpg')
    im = im.convert('RGB')
    im_input = np.array(im, dtype=np.uint8)

    h, w, _ = im_input.shape
    height, width = 2 * h-10, 2 * w-10
    im_src = np.zeros([height, width, _])
    src_map = np.zeros([height, width]).astype(np.bool)

    seam_map = SeamMap(height, width)
    pos = [(0, 0), (0, w-10), (h-10, 0), (h-10, w-10)]
    for offset in pos: 
        # error_region = get_error_region(im_src, src_map, seam_map, (h // 4, w // 4))
        # debug_error_region(im_src, error_region)
        # offset = get_offset_entire_matching(im_src, src_map, im_input, None, False)
        print(offset)
        patch_fitting(im_src, src_map, im_input, offset, seam_map)
        show_im = Image.fromarray(im_src.astype(np.uint8))
        show_im.save('save.jpg')
    debug_cut(im_src, seam_map).save('output/cut.jpg')

    error_region = get_error_region(im_src, src_map, seam_map, (h // 4, w // 4))

    
    offset = get_offset_entire_matching(im_src, src_map, im_input, error_region)
    patch_fitting(im_src, src_map, im_input, offset, seam_map)
    show_im = Image.fromarray(im_src.astype(np.uint8))
    show_im.save('save.jpg')
    debug_cut(im_src, seam_map).save('output/cut2.jpg')

def get_offset_auto(im_src, src_map, im_input, error_region: Region, region_size, Gamma):
    height, width = src_map.shape
    gamma = src_map.astype(np.int).sum() / (height * width)
    if gamma > Gamma:
        return get_offset_subpatch_matching(im_src, src_map, im_input, error_region, region_size)
    else: 
        return get_offset_entire_matching(im_src, src_map, im_input, error_region)


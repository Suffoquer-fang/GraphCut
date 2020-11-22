import numpy as np
import networkx as nx 
from PIL import Image

def show(a):
    print(a.astype(np.uint8))

def build_boarder(height, width, dir):
    if dir == 0:
        return [(0, i) for i in range(width)]
    elif dir == 1:
        return [(i, width-1) for i in range(height)]
    elif dir == 2:
        return [(height-1, i) for i in range(width)]
    else:
        return [(i, 0) for i in range(height)]

def build_graph(im_src, im_dst, force_src, force_dst, overlap_map):
    im_diff = (im_src - im_dst) ** 2
    im_diff = np.sum(im_diff, axis = 2)

    height, width = im_src[:, :, 0].shape

    s = 0
    f = height * width + 1

    ret_edges = []
    for i in range(height):
        for j in range(width):
            if overlap_map[i, j] == 0: continue
            u = i * width + j + 1 
            neibors = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
            for x, y in neibors:
                if 0 <= x and x < height and 0 <= y and y < width and overlap_map[x, y] == 1:
                    
                    v = x * width + y + 1
                    M = im_diff[i, j] + im_diff[x, y]
                    ret_edges.append((u, v, M))
            if force_src[i, j] == 1 and force_dst[i, j] == 0:
                ret_edges.append((s, u, float("inf")))
            if force_src[i, j] == 0 and force_dst[i, j] == 1:
                ret_edges.append((f, u, float("inf")))
     
    return s, f, ret_edges


def patch_fitting(im_src, im_dst, force_src, force_dst, overlap_map):
    s, f, edges = build_graph(im_src, im_dst, force_src, force_dst, overlap_map)
    G = nx.Graph()
    G.add_weighted_edges_from(edges)

    height, width = im_src[:, :, 0].shape
    if G.has_node(s) and G.has_node(f):
        _, partion = nx.minimum_cut(G, s, f, 'weight')
        left, right = partion
        print('done')
        im_tmp = im_dst.copy()
        for idx in left:
            if idx == s: continue
            x, y = (idx - 1) // width, (idx - 1) % width
            print(x, y)
            im_tmp[x, y] = im_src[x, y]
        
        im_src[:, :, :] = im_tmp[:, :, :]
        
    else: 
        im_src[:, :, :] = im_dst[:, :, :]
    
    # im_out = im_src.copy()
    # im_src[:, :, :] = im_dst[:, :, :]
    
    


def prepare_patch_fitting(im, im_input, im_mask, offset):
    # height, width, _ = im.shape
    h, w, _ = im_input.shape

    # im_mask = (im != 0)
    im_mask_input = np.zeros(im_mask.shape)
    im_mask_input[offset[0]:offset[0] + h, offset[1]: offset[1] + w] = 1

    overlap_map = im_mask * im_mask_input
    # print(im_mask_input.astype(np.uint8))
    # print(overlap_map.astype(np.uint8))

    force_src_map = get_force_map(im_mask, overlap_map)
    force_dst_map = get_force_map(im_mask_input, overlap_map)
    # print('*' * 50)
    # show(force_src_map)
    # print()
    # show(force_dst_map)

    print('im_mask')

    show(im_mask)
    print('overlap')
    show(overlap_map)
    print('im_mask_input')
    show(im_mask_input)

    force_src_map = force_src_map[offset[0]:offset[0] + h, offset[1]:offset[1] + w]
    force_dst_map = force_dst_map[offset[0]:offset[0] + h, offset[1]:offset[1] + w]
    overlap_map = overlap_map[offset[0]:offset[0] + h, offset[1]:offset[1] + w]

    overlap_map = overlap_map.astype(np.uint8)
    print('*' * 50)
    show(force_src_map)
    print()
    show(force_dst_map)

    im_src = im[offset[0]:offset[0] + h, offset[1]:offset[1] + w]
    im_dst = im_input

    return im_src, im_dst, force_src_map, force_dst_map, overlap_map
    

def get_force_map(mask, overlap):
    ret_map = np.zeros(mask.shape)
    coords = np.where(overlap > 0)
    h, w = overlap.shape
    for idx in range(len(coords[0])):
        i, j = coords[0][idx], coords[1][idx]
        neibors = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
        for x, y in neibors:
            if 0 <= x and x < h and 0 <= y and y < w:
                if mask[x, y] == 1 and overlap[x, y] == 0:
                    ret_map[i, j] = 1
    return ret_map


if __name__ == '__main__':
    
    im = Image.open('data/strawberries2.jpg')
    # im.show()
    im = im.convert('RGB')
    im_input = np.array(im, dtype=np.uint8)
    # print(im)
    # im_input = im_input[:, :]
    h, w, _ = im_input.shape
    overlap = 40

    im = np.zeros([2 * h - 20, 2 * w - overlap, _])
    im_mask = np.zeros([2 * h - 20, 2 * w - overlap])
    offset = [0, 0]


    

    im_src, im_dst, force_src_map, force_dst_map, overlap_map = prepare_patch_fitting(im, im_input, im_mask, offset)
    patch_fitting(im_src, im_dst, force_src_map, force_dst_map, overlap_map)

    im_mask[:h, :w] = 1
    offset = [0, w-overlap]

    im_src, im_dst, force_src_map, force_dst_map, overlap_map = prepare_patch_fitting(im, im_input, im_mask, offset)
    patch_fitting(im_src, im_dst, force_src_map, force_dst_map, overlap_map)

    im_mask[:h, :] = 1
    offset = [h-20, 0]

    im_src, im_dst, force_src_map, force_dst_map, overlap_map = prepare_patch_fitting(im, im_input, im_mask, offset)
    patch_fitting(im_src, im_dst, force_src_map, force_dst_map, overlap_map)

    im_mask[:, :w] = 1
    offset = [h-20, w-overlap]

    im_src, im_dst, force_src_map, force_dst_map, overlap_map = prepare_patch_fitting(im, im_input, im_mask, offset)
    patch_fitting(im_src, im_dst, force_src_map, force_dst_map, overlap_map)

    show_im = Image.fromarray(im.astype(np.uint8))
    show_im.show()
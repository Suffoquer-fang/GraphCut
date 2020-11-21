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

def build_graph(im_src, im_dst, force_src, force_dst):
    im_diff = (im_src - im_dst) ** 2
    im_diff = np.sum(im_diff, axis = 2)

    height, width = im_src[:, :, 0].shape

    s = 0
    f = height * width + 1

    ret_edges = []
    for i in range(height):
        for j in range(width):
            u = i * width + j + 1 
            neibors = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
            for x, y in neibors:
                if 0 <= x and x < height and 0 <= y and y < width:
                    
                    v = x * width + y + 1
                    M = im_diff[i, j] + im_diff[x, y]
                    ret_edges.append((u, v, M))
    
    for di in force_src:
        nodes = build_boarder(height, width, di)
        for i, j in nodes:
            ret_edges.append((s, i * width + j + 1, float("inf")))
    
    for di in force_dst:
        nodes = build_boarder(height, width, di)
        for i, j in nodes:
            ret_edges.append((f, i * width + j + 1, float("inf")))
    
    return s, f, ret_edges


def patch_fitting(im_src, im_dst, force_src, force_dst):
    s, f, edges = build_graph(im_src, im_dst, force_src, force_dst)
    G = nx.Graph()
    G.add_weighted_edges_from(edges)

    _, partion = nx.minimum_cut(G, s, f, 'weight')
    _, right = partion

    height, width = im_src[:, :, 0].shape
    im_out = im_src.copy()
    for idx in right:
        if idx == f: continue
        x, y = (idx - 1) // width, (idx - 1) % width
        # print(x, y)
        im_out[x, y] = im_dst[x, y]
    return im_out


def prepare_patch_fitting(im, im_input, im_mask, offset):
    # height, width, _ = im.shape
    h, w = im_input.shape

    # im_mask = (im != 0)
    im_mask_input = np.zeros(im_mask.shape)
    im_mask_input[offset[0]:offset[0] + h, offset[1]: offset[1] + w] = 1

    overlap_map = im_mask * im_mask_input
    # print(im_mask_input.astype(np.uint8))
    # print(overlap_map.astype(np.uint8))

    force_src_map = get_force_map(im_mask, overlap_map)
    force_dst_map = get_force_map(im_mask_input, overlap_map)
    print('*' * 50)
    show(force_src_map)
    print()
    show(force_dst_map)

    force_src_map = force_src_map[offset[0]:offset[0] + h, offset[1]:offset[1] + w]
    force_dst_map = force_dst_map[offset[0]:offset[0] + h, offset[1]:offset[1] + w]
    overlap_map = overlap_map[offset[0]:offset[0] + h, offset[1]:offset[1] + w]

    print('*' * 50)
    show(force_src_map)
    print()
    show(force_dst_map)

    im_src = im[offset[0]:offset[0] + h, offset[1]:offset[1] + w] * overlap_map
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
    
    im = Image.open('./akeyboard_small.jpg')
    # im.show()
    im = im.convert('RGB')
    im = np.array(im, dtype=np.uint8)
    # print(im)
    print(im.shape)
    h, w, _ = im.shape

    # overlap = 70
    # output_im = np.zeros([2 * h - overlap, w, _])
    # output_im[0:h, :] = im 
    # output_im[h-overlap:] = im 

    # im_src = im[-overlap:, :]
    # im_dst = im[:overlap, :]
    # output_im[h-overlap:h, :] = patch_fitting(im_src, im_dst, [0], [2])

    # show_im = Image.fromarray(output_im.astype(np.uint8))
    # show_im.show()

    # overlap = 20
    # output_im = np.zeros([h, 2 * w - overlap, _])
    # output_im[:, :w] = im 
    # output_im[:, w-overlap:] = im 

    # im_src = im[:, -overlap:]
    # im_dst = im[:, :overlap]
    # output_im[:, w-overlap:w] = patch_fitting(im_src, im_dst, [3], [1])

    # show_im = Image.fromarray(output_im.astype(np.uint8))
    # show_im.show()

    im = np.zeros([20, 20])
    im[:12, :12] = 2
    im_mask = (im != 0)
    im = im.astype(np.uint8)
    im_mask = im_mask.astype(np.uint8)
    
    print(im)
    print(im_mask)

    im_input = np.ones([12, 10]) * 3
    # im_input[:, :] = 3
    offset = [0, 4]

    prepare_patch_fitting(im, im_input, im_mask, offset)
    # print(prepare_patch_fitting(im, im_input, im_mask, offset))
    # print(np.where(im_mask > 0))
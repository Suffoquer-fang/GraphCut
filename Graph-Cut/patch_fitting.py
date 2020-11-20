import numpy as np
import networkx as nx 
from PIL import Image

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


def prepare_patch_fitting(im, im_input, offset):
    height, width, _ = im.shape
    h, w, _ = im_input.shape

    



if __name__ == '__main__':
    
    im = Image.open('./akeyboard_small.jpg')
    # im.show()
    im = im.convert('RGB')
    im = np.array(im, dtype=np.uint8)
    # print(im)
    print(im.shape)
    h, w, _ = im.shape

    overlap = 70
    output_im = np.zeros([2 * h - overlap, w, _])
    output_im[0:h, :] = im 
    output_im[h-overlap:] = im 

    im_src = im[-overlap:, :]
    im_dst = im[:overlap, :]
    output_im[h-overlap:h, :] = patch_fitting(im_src, im_dst, [0], [2])

    show_im = Image.fromarray(output_im.astype(np.uint8))
    show_im.show()
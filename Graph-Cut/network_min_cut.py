import networkx as nx 
from min_cut import *
import time


if __name__ == '__main__':
    G = nx.Graph()
    G.add_edge(1, 2, capacity=20)
    G.add_edge(1, 3, capacity=10)
    G.add_edge(2, 3, capacity=5)
    G.add_edge(2, 4, capacity=10)
    G.add_edge(3, 4, capacity=20)

    # print(G.edges())

    # min_cut = nx.max_flow_min_cost(G, 1, 4)
    # print(min_cut)
    # print(nx.minimum_cut(G, 1, 4))
    
    # G = {}
    # addDoubleEdge(G, 1, 2, 20, 20)
    # addDoubleEdge(G, 1, 3, 10, 10)
    # addDoubleEdge(G, 2, 3, 5, 5)
    # addDoubleEdge(G, 2, 4, 10, 10)
    # addDoubleEdge(G, 3, 4, 20, 20)
    # mincut = minCut(G, 1, 4)
    # print(mincut)
    # print(sum([G[i][j] for i, j in mincut]))

    im = Image.open('./akeyboard_small.jpg')
    im.show()
    im = im.convert('RGB')
    im = np.array(im, dtype=np.uint8)
    # print(im)
    print(im.shape)



    overlap = 20

    im_src = im[:, -overlap:]
    print(im_src.shape)
    im_dst = im[:, :overlap]
    print(im_dst.shape)

    st = time.time()

    out = patchFitting(im_src, im_dst)

    print('time1:', time.time() - st)

    st = time.time()

    G = nx.Graph()

    im_diff = (im_src - im_dst) ** 2
    im_diff = np.sum(im_diff, axis = 2)

    height, width = im_src[:, :, 0].shape
    # height, width = im_src.shape
    f = height * width + 1
    for i in range(height):
        s = i * width + 1
        t = (i + 1) * width
        G.add_edge( 0, s, capacity=float("inf"))
        G.add_edge( t, f, capacity=float("inf"))


    for i in range(height):
        for j in range(width):
            s = i * width + j + 1 
            # addDoubleEdge(G, 0, s, float("inf"), float("inf"))
            # addDoubleEdge(G, s, f, float("inf"), float("inf"))
            neibors = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
            for x, y in neibors:
                if 0 <= x and x < height and 0 <= y and y < width:
                    
                    t = x * width + y + 1
                    M = im_diff[i, j] + im_diff[x, y]
                    G.add_edge( s, t, capacity=M)
    
    _, par = nx.minimum_cut(G, 0, f)
    le, ri = par 

    print('time2:', time.time() - st)

    print(out == ri)
    
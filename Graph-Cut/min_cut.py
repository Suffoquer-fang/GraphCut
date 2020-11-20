import numpy as np 
from PIL import Image
import time

def addEdge(G, u, v, w):
    if u not in G:
        G[u] = {}
    G[u][v] = w

def addDoubleEdge(G, u, v, w1, w2 = 0):
    addEdge(G, u, v, w1)
    addEdge(G, v, u, w2)

def BFS(G, src, dst):
    n = len(G)
    queue = [src]
    prev = {}
    flow = {}
    prev[src] = 0
    flow[src] = None
    while len(queue) > 0:
        cur = queue.pop(0)
        if cur == dst:
            break

        for v in G[cur]:
            if v not in prev and G[cur][v] > 0 and v not in queue:
                queue.append(v)
                flow[v] = G[cur][v] if flow[cur] is None else min(flow[cur], G[cur][v])
                prev[v] = cur 
        

    if dst not in prev:
        return -1, None
    else:
        ret_path = []
        i = dst
        while i > 0:
            ret_path.append(i)
            i = prev[i]

        return flow[dst], ret_path



def maxFlow(G, src, dst):
    ret = 0
    while True:
        flow, path = BFS(G, src, dst)
        if flow == -1:
            return ret 
        
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            G[v][u] -= flow 
            G[u][v] += flow
        
        ret += flow
    return ret 

def minCut(G, src, dst):
    ret_cut = []
    maxFlow(G, src, dst)
    for u in G:
        for v in G[u]:
            if G[u][v] == 0:
                if (v, u) not in ret_cut:
                    ret_cut.append((u, v))

    return set(ret_cut)



def patchFitting(im_src, im_dst):
    st = time.time()
    G = {}
    im_diff = (im_src - im_dst) ** 2
    im_diff = np.sum(im_diff, axis = 2)

    height, width = im_src[:, :, 0].shape
    # height, width = im_src.shape
    f = height * width + 1
    for i in range(height):
        s = i * width + 1
        t = (i + 1) * width
        addDoubleEdge(G, 0, s, float("inf"), float("inf"))
        addDoubleEdge(G, t, f, float("inf"), float("inf"))


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
                    addDoubleEdge(G, s, t, M, M)
    
    im_out = im_src.copy()

    min_cut = minCut(G, 0, f)
    print(time.time() - st)
    # print(min_cut)
    for i, j in min_cut:
        del G[i][j]
        del G[j][i]
    

    right = tarjan(G, f)
    # left = tarjan(G, 0)
    # print(set(left) & set(right))
    print('done tarjan')

    return right
    # for i in range(height):
    #     for j in range(width):
    #         s = i * width + j + 1
    #         if s in right:
    #             im_out[i, j] = im_dst[i, j]

    # return im_out

def tarjan(G, s):
    # ret = set()
    vis = [0] * (s + 1)
    queue = [s]
    while len(queue) > 0:
        cur = queue.pop(0)
        vis[cur] = 1
        for v in G[cur]:
            if vis[v] == 0 and v not in queue:
                queue.append(v)
    ret = [i for i in range(s+1) if vis[i] == 1]
    return set(ret)



# im_out = patchFitting(im_src, im_dst)

# h, w, _ = im.shape
# new_im = np.zeros([h, 2 * w - overlap, 3])
# print(new_im.shape)
# new_im[:, :im.shape[1]] = im
# new_im[:, im.shape[1]-overlap:] = im 

# show_im = Image.fromarray(new_im.astype(np.uint8))
# show_im.show()


# new_im[:, im.shape[1]-overlap:im.shape[1]] = im_out
# show_im = Image.fromarray(new_im.astype(np.uint8))
# show_im.show()
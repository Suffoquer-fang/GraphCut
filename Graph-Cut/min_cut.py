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
            if v not in prev and G[cur][v] > 0:
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
                ret_cut.append((u, v))

    return ret_cut

G = {}
addDoubleEdge(G, 1, 2, 40)
addDoubleEdge(G, 1, 4, 20)
addDoubleEdge(G, 2, 3, 30)
addDoubleEdge(G, 2, 4, 20)
addDoubleEdge(G, 3, 4, 10)


print(minCut(G, 1, 4))
         
    
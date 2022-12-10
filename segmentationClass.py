"""
CSCI 332
Esther Lyon Delsordo
Software Assignment #2
Due December 10, 2022

Implement Ford-Fulkerson Alg and use it to implement the image
segmentation algorithm from Kelinberg-Tardos 7.10
"""

import numpy as np

"""
Vertex class to build graph nodes
"""


class vertex:
    def __init__(
        self, key, pos, rgb, edgesIn, edgesOut, capacities, dim, flows, fg, bg
    ):
        self.key = key
        self.pos = pos
        self.rgb = rgb
        self.edgesIn = []  # keys of nodes this node has edges from
        self.edgesOut = []  # keys of nodes this node points to
        self.capacities = (
            {}
        )  # key = key of other node in edge, value = [capacity of edge in, capacity of edge out]
        self.dim = dim  # Image dimension (pixels), square images only
        self.flows = (
            {}
        )  # key = key of other node in edge, value = [flow on edge in, flow on edge out]
        self.fg = False
        self.bg = False


"""
Segmentation Class
"""


class segmentationClass:
    # initialize a class instance
    def __init__(self, p0, x_a, x_b):
        self.p0 = p0
        self.x_a = x_a
        self.x_b = x_b

    """
    Input: I is an NxNx3 numpy array representing a color (RGB) image. Each pixel
    intensity will be have an integer value between 0 and 255.
    ● Output: L is an NxNx1 numpy array of binary values representing whether each pixel
    in the image is in the foreground (value of 1) or in the background (value of 0). So, if
    pixel at row (i,j) is in the foreground then L[i,j] = 1, and if it is in the background we
    have L[i,j] = 0.
    """

    def segmentImage(I):
        N = len(I.shape[0])  # get image dimension
        G = []  # initialize graph
        Gf = []  # initialize residual graph

        # build graph
        # make each pixel a vertex and add to graph
        key = 0
        for i in range(N):
            for j in range(N):
                newNode = vertex(key, [i, j], I[i][j], dim=N)
                G = np.append(newNode)
                key += 1

        # Connect Graph pixel nodes to their neighbors
        for node in G:
            for neighbor in G[node.key :]:
                d = np.sqrt(((node.pos - neighbor.pos) ** 2).sum())
                if d < 2:
                    # make edge from neighbor to node
                    node.edgesIn.append(neighbor.key)

                    # make edge from node to neighbor
                    node.edgesOut.append(neighbor.key)

                    # set capacities of both edges to p0
                    node.capacities[neighbor.key] = [segmentationClass.p0] * 2

                    # set all new edge flows to zero
                    node.flows[neighbor.key] = [0] * 2

        # Add source and sink nodes to graph
        source = vertex(key, dim=N)
        sink = vertex(key + 1, dim=N)
        G.append(source)
        G.append(sink)

        # connect source and sink to all pixel nodes
        for node in G[:-2]:
            # connect source and sink to all nodes
            source.edgesOut.append(node.key)
            sink.edgesIn.append(node.key)

            # set capacities on edges from source and to sink
            source.capacities[node.key] = [0, segmentationClass.foregroundProb(node, G)]
            source.capacities[node.key] = [segmentationClass.backgroundProb(node, G), 0]

            # set flows to zero on all new edges
            source.flows[node.key] = [0] * 2
            sink.flows[node.key] = [0] * 2

        # call FF on graph

        # classify nodes as FG or BG by checking
        # if each node is connected to FG, if not, they are in BG

        # return N*N*1 array, last dim is class 0 (BG) or 1 (FG)
        return -1

    # Ford Fulkerson (FF) Alg for Min Cut
    # remove edges from graph to produce min cut
    def findMinCut(G):
        # all flows in G start at 0
        # while there is an s-t path in residual graph G'
        # (use bfs or dfs to find an s-t path)
        # let P be a simple s-t path in Gf
        # f' = augment(f,P)
        # update f to be f'
        # update the residual graph Gf to be Gf'
        # endwhile
        # return f
        return -1

    # BFS for FF
    # This makes my FF the Edmonds-Karp variation of the algorithm
    # reference: https://www.geeksforgeeks.org/minimum-cut-in-a-directed-graph/?ref=rp
    # parent is a list to store the s-t path
    def bfs(Gf, s, t, parent):
        # list to store visited nodes
        visited = [False] * len(Gf)

        # queue for BFS
        queue = []

        # add source to queue and mark as visited
        queue.append(s)
        visited[s.key] = True

        # BFS loop
        while queue:
            # dequeue first node in queue
            u = queue.pop(0)

            # Get all adjacent vertices of the dequeued vertex u
            # If a adjacent has not been visited, then mark it
            # visited and enqueue it
            for node in s.edgesOut:
                if visited[node] == False and s.capacities[node][1] > 0:
                    queue.append(node)
                    visited[node.key] = True
                    parent[node.key] = u

        return True if visited[t] else False

    # FIXME remove?
    # DFS to traverse the original graph?
    def dfs(G):
        return -1

    # The probability a pixel node is in the foreground
    # Defined as 442 - Euclidean dist between RBG vals of x and x_a
    def foregroundProb(x, G):
        N = np.sqrt(len(G) - 2)
        Grid = np.reshape(G, (-1, N))
        x_a_key = Grid[segmentationClass.x_a[0], segmentationClass.x_a[1]].key
        d = np.round(np.sqrt(((Grid[x_a_key].rgb - x.rgb) ** 2).sum()))
        return 442 - d

    # The probability a pixel node is in the background
    # Defined as 442 - Euclidean dist between RBG vals of x and x_b
    def backgroundProb(x, G):
        N = np.sqrt(len(G) - 2)
        Grid = np.reshape(G, (-1, N))
        x_b_key = Grid[segmentationClass.x_b[0], segmentationClass.x_b[1]].key
        d = np.round(np.sqrt(((Grid[x_b_key].rgb - x.rgb) ** 2).sum()))
        return 442 - d

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
    def __init__(self, key, pos, rgb, edgesIn, edgesOut, capacities, dim, flows):
        self.key = key
        self.pos = pos
        self.rgb = rgb
        self.edgesIn = []
        self.edgesOut = []
        self.capacities = {}
        self.dim = dim  # Image dimension (pixels), square images only
        self.flows = [dim * (dim + 2)]


"""
Segmentation Class
"""


class segmentationClass:
    # initialize a class instance
    def __init__(self, p0, x_a, x_b):
        self.p0 = p0
        self.x_a = x_a
        self.x_b = x_b

    # Initialize graph and residual graph
    G = []
    Gf = []

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
                newNode = vertex(key, [i, j], I[i][j])
                G = np.append(G, newNode)
                key += 1

        # Connect Graph pixel nodes to their neighbors
        for node in G:
            for neighbor in G[node.key :]:
                d = np.sqrt(((node.pos - neighbor.pos) ** 2).sum())
                if d < 2:
                    node.edgesIn.append(node.edgesIn, neighbor.key)
                    node.edgesOut.append(node.edgesOut, neighbor.key)

        # Add source and sink nodes to graph
        source = vertex(key)
        sink = vertex(key + 1)
        G.append(G, source)
        G.append(G, sink)

        # connect source and sink to all pixel nodes
        for node in G[:-2]:
            source.edgesOut.append(source.edgesOut, node.key)
            sink.edgesIn.append(sink.edgesIn, node.key)

        # call FF on graph

        # classify nodes as FG or BG by checking FG and BG probs

        # return N*N*1 array, last dim is class 0 (BG) or 1 (FG)
        return -1

    # Ford Fulkerson (FF) Alg for Max Flow
    def findMaxFlow(G):
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

    # augment a path in the residual graph for FF
    def augment(path, flow):
        b = segmentationClass.bottleneck(path, flow)
        # for each edge (u,v) in path P
        # if e = (u,v) is a forward edge
        # f(e) += b in G
        # if (u,v) is a backward edge
        # let e = (v,u)
        # f(e) -= b in G
        # endfor
        # return flow
        return -1

    # bottleneck function for augment function
    # finds the minimum residual capacity on any edge in an s-t path
    def bottleneck(path, flow):
        # traverse all edges in path, find the smallest capacity, c
        # return c - flow
        return -1

    # determines capacity between pixel nodes
    def neighborCapacity(u, v):
        # calculate Euclidean distance from u to v
        d = np.sqrt((v[0] - u[0]) ** 2 + (v[1] - u[1]) ** 2)
        if d < 2:
            return segmentationClass.p0
        else:
            return 0

    # FIXME
    # The probability a pixel is in the foreground
    # Defined as 442 - Euclidean dist between RBG vals of x and x_a
    def foregroundProb(x):
        # calculate Euclidean distance from x to x_a
        d = np.sqrt(
            (segmentationClass.x_a[0] - x[0]) ** 2
            + (segmentationClass.x_a[1] - x[1]) ** 2
        )
        return 442 - d

    # FIXME
    # The probability a pixel is in the background
    # Defined as 442 - Euclidean dist between RBG vals of x and x_b
    def backgroundProb(x):
        # calculate Euclidean distance from x to x_b
        d = np.sqrt(
            (segmentationClass.x_b[0] - x[0]) ** 2
            + (segmentationClass.x_b[1] - x[1]) ** 2
        )
        return 442 - d

    # DFS Algorithm for FF
    # Faster than BFS and goes to end of a path first
    # This makes more sense for FF
    # This implementation is based on one from Fahadul Shadhin via Medium
    # URL: https://medium.com/geekculture/depth-first-search-dfs-algorithm-with-python-2809866cb358
    # original version is for adjacency list graph rep FIXME?
    # adjacency matrix version: https://www.geeksforgeeks.org/implementation-of-dfs-using-adjacency-matrix/
    def dfs(graph, source, visited, dfs_traversal):
        if source not in visited:
            dfs_traversal.append(source)
            visited.add(source)

        for neighbor_node in graph[source]:
            segmentationClass.dfs(graph, neighbor_node, visited, dfs_traversal)

        return dfs_traversal

    # main function with driver code for dfs FIXME
    def main():
        visited = set()
        dfs_traversal = list()

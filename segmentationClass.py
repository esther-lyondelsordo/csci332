"""
CSCI 332
Esther Lyon Delsordo
Software Assignment #2
Due December 10, 2022

Implement Ford-Fulkerson Alg and use it to implement the image
segmentation algorithm from Kelinberg-Tardos 7.10
"""

import numpy as np


class segmentationClass:
    # Function to return a class instance
    def segmentationClass(self, p0, x_a, x_b):
        self.p0 = p0  # penalty for two pixels being neighbors
        self.x_a = x_a  # coordinates of a chosen foreground pixel
        self.x_b = x_b  # coordinates of a chosen background pixel
        return segmentationClass()

    # init class, do I need this? FIXME
    def __init__(self, p0, x_a, x_b):
        self.p0 = p0
        self.x_a = x_a
        self.x_b = x_b

    """
    Input: I is an NxNx3 numpy array representing a color (RGB) image. Each pixel
    intensity will be have an integer value between 0 and 255.
    ● Output: L is an NxNx1 numpy array of binary values representing whether each pixel
    in the image is in the foreground (value of 1) or in the background (value of 0). So, if
    pixel at row (𝑖,𝑗) is in the foreground then 𝐿[𝑖,𝑗] = 1, and if it is in the background we
    have 𝐿[𝑖,𝑗] = 0.
    """

    def segmentImage(Img):
        # build graph
        # call FF on graph
        #
        return -1

    # make the graph with source and sink nodes from the pixel arrays
    # I used the adjacency list representation FIXME ?
    def makeGraph(Img):
        # set all flows to zero initially?
        # return G
        N = len(Img[0]) + 2  # add two for source and sink nodes
        graph = {}
        # make edges from source to all pixels
        # make edges from sink to all pixels
        # make edges between all pixels and their L,R,U,D neighbors
        return graph

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

    # make residual graph for FF
    def makeResidualGraph(G):

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

    def neighborPenalty(u, v):
        # calculate Euclidean distance from u to v
        d = np.sqrt((v[0] - u[0]) ** 2 + (v[1] - u[1]) ** 2)
        if d < 2:
            return segmentationClass.p0
        else:
            return 0

    # The probability a pixel is in the foreground
    # Defined as 442 - dist from x to x_a
    def foregroundProb(x):
        # calculate Euclidean distance from x to x_a
        d = np.sqrt(
            (segmentationClass.x_a[0] - x[0]) ** 2
            + (segmentationClass.x_a[1] - x[1]) ** 2
        )
        return 442 - d

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

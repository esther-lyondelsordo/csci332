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
        self,
        key,
        dim,
        pos=np.array([]),
        rgb=np.array([]),
        edgesIn=[],
        edgesOut=[],
        capacities={},
    ):
        self.key = key
        self.dim = dim  # Image dimension (pixels), square images only
        self.pos = pos
        self.rgb = rgb
        self.edgesIn = edgesIn  # keys of nodes this node has edges from
        self.edgesOut = edgesOut  # keys of nodes this node points to
        self.capacities = capacities  # key = key of other node in edge,
        # value = [capacity of edge in to other node, capacity of edge out of other node]


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

    def segmentImage(self, I):
        N = I.shape[0]  # get image dimension
        G = []  # initialize graph
        Gf = []  # initialize residual graph

        # build graph
        # make each pixel a vertex and add to graph
        k = 0
        for i in range(N):
            for j in range(N):
                print("k: ", k)
                newNode = vertex(
                    key=k,
                    dim=N,
                    pos=np.array([i, j]),
                    rgb=(np.rint(I[i][j])).astype(
                        int
                    ),  # round float rgb vals to nearest int
                )
                G.append(newNode)
                k += 1

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
                    node.capacities[neighbor.key] = [self.p0] * 2

        # Add source and sink nodes to graph
        print("k: ", k)
        source = vertex(key=k, dim=N)
        k += 1
        print("k: ", k)
        sink = vertex(key=k, dim=N)
        G.append(source)
        G.append(sink)

        # connect source and sink to all pixel nodes
        for node in G[:-2]:
            # connect source and sink to all nodes
            source.edgesOut.append(node.key)
            sink.edgesIn.append(node.key)

            # set capacities on edges from source and to sink
            source.capacities[node.key] = [
                self.foregroundProb(node, G),
                0,
            ]
            sink.capacities[node.key] = [
                0,
                self.backgroundProb(node, G),
            ]

        # Initialize redsidual graph
        Gf = G
        s = Gf[-2]
        t = Gf[-1]

        # call FF on residual graph
        Gf = self.findMinCut(Gf, s, t)

        # classify nodes as FG or BG by checking
        # if each node is connected to source (FG), if not, they are in BG
        # return N*N*1 array, last dim is class 0 (BG) or 1 (FG)
        outputImage = [[0] * N] * N
        for node in Gf[:-2]:
            if s.capacities[node][0] == 0:
                outputImage[node.pos[0], node.pos[1]] = 1

        return outputImage

    # Ford Fulkerson (FF) Alg for Min Cut
    # remove edges from graph to produce min cut
    def findMinCut(self, Gf, source, sink):

        # Store the path from BFS
        parent = [-1] * len(Gf)

        # All flows start at zero
        maxFlow = 0

        # Use BFS to add flow while there is an s-t path
        while self.BFS(Gf, source, sink, parent):

            # Find max flow through the augmenting path
            # This is equal to the min residual capacity of the
            # edges along the path
            # start augmenting path flow at infinity to make
            # sure any existing capacity is less than it
            augmenting_flow = float("Inf")
            s = sink.key  # start from sink
            while s != source:
                # update aug flow to min of edge from parent to s and aug flow
                augmenting_flow = min(augmenting_flow, Gf[parent[s]].capacities[s][1])

                # move to next node in path
                s = parent[s]

            # Add augmenting flow to total max flow
            maxFlow += augmenting_flow

            # Update residual capacities of edges along the path
            # reverse edges if needed
            curr = sink
            while curr != source:
                prev = parent[curr]
                # decrement resid cap from parent to current node
                Gf[prev].capacities[curr][0] -= augmenting_flow
                Gf[curr].capacities[prev][1] -= augmenting_flow

                # increment resid cap from current to parent
                Gf[prev].capacities[curr][1] += augmenting_flow
                Gf[curr].capacities[prev][0] += augmenting_flow

                curr = parent[curr]

        return Gf

    # BFS for FF
    # This makes my FF the Edmonds-Karp variation of the algorithm
    # reference: https://www.geeksforgeeks.org/minimum-cut-in-a-directed-graph/?ref=rp
    # parent is a list to store the s-t path
    def bfs(self, Gf, s, t, parent):
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

    # The probability a pixel node is in the foreground
    # Defined as 442 - Euclidean dist between RBG vals of x and x_a
    def foregroundProb(self, x, G):
        N = np.sqrt(len(G) - 2)
        Grid = np.reshape(G, (-1, N))
        x_a_key = Grid[self.x_a[0], self.x_a[1]].key
        d = np.round(np.sqrt(((Grid[x_a_key].rgb - x.rgb) ** 2).sum()))
        return 442 - d

    # The probability a pixel node is in the background
    # Defined as 442 - Euclidean dist between RBG vals of x and x_b
    def backgroundProb(self, x, G):
        N = np.sqrt(len(G) - 2)
        Grid = np.reshape(G, (-1, N))
        x_b_key = Grid[self.x_b[0], self.x_b[1]].key
        d = np.round(np.sqrt(((Grid[x_b_key].rgb - x.rgb) ** 2).sum()))
        return 442 - d

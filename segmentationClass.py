# segmentationClass.py
"""
CSCI 332
Esther Lyon Delsordo
Software Assignment #2
Due December 10, 2022

Implement Ford-Fulkerson Alg and use it to implement the image
segmentation algorithm from Kelinberg-Tardos 7.10
"""

import numpy as np
import copy

"""
Vertex class to build graph nodes and store their associated data
"""


class vertex:
    def __init__(
        self,
        key,
        dim,
        pos=None,
        rgb=None,
        edgesIn=None,
        edgesOut=None,
        capacities=None,
        parent=None,
    ):
        self.key = key

        # Image dimension (pixels), square images only
        self.dim = dim

        self.pos = np.array([]) if pos is not None else None
        self.rgb = np.array([]) if rgb is not None else None

        # keys of nodes this node has edges from
        self.edgesIn = [] if edgesIn is not None else None

        # keys of nodes this node points to
        self.edgesOut = [] if edgesOut is not None else None

        # key = key of other node in edge,
        # value = [capacity of edge in to other node, capacity of edge out of other node]
        self.capacities = {} if capacities is not None else None

        # Store the path from BFS by storing the parent of each node
        self.parent = parent if parent is not None else None


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
                newNode = vertex(key=k, dim=N, pos=np.array([]), rgb=np.array([]),)
                newNode.pos = np.array([i, j])
                newNode.rgb = I[i][j].astype(
                    int
                )  # .astype(int),  # change float rgb vals to int
                G.append(newNode)
                k += 1

        # Connect Graph pixel nodes to their neighbors
        for node in G:
            node.edgesIn = []
            node.edgesOut = []
            node.capacities = {}
            for neighbor in G:
                # If the Euclidean distance between nodes is
                # less than 2, the edge weight is p0
                # and zero otherwise
                d = np.sqrt(((node.pos - neighbor.pos) ** 2).sum())
                if d < 2:
                    # make edge from neighbor to node
                    node.edgesIn.append(neighbor.key)

                    # make edge from node to neighbor
                    node.edgesOut.append(neighbor.key)

                    # set capacities of both edges to p0
                    node.capacities[neighbor.key] = [self.p0] * 2

        # Add source and sink nodes to graph
        sink = vertex(key=k, dim=N)
        k += 1
        source = vertex(key=k, dim=N)
        G.append(sink)
        G.append(source)

        # Initialize source and sink attributes
        s = len(G) - 1
        t = len(G) - 2
        G[s].edgesIn = []
        G[s].edgesOut = []
        G[s].capacities = {}
        G[t].edgesIn = []
        G[t].edgesOut = []
        G[t].capacities = {}

        # connect source and sink to all pixel nodes
        for node in G[:-2]:
            # connect source and sink to all nodes
            G[s].edgesOut.append(node.key)
            node.edgesIn.append(G[s].key)
            G[t].edgesIn.append(node.key)
            node.edgesOut.append(G[t].key)

            # set capacities on edges from source and to sink
            G[s].capacities[node.key] = [
                self.foregroundProb(node.key, G),
                0,
            ]
            G[t].capacities[node.key] = [
                0,
                self.backgroundProb(node.key, G),
            ]
            node.capacities[G[s].key] = [0, self.backgroundProb(node.key, G)]
            node.capacities[G[t].key] = [self.foregroundProb(node.key, G), 0]

        # Initialize redsidual graph
        Gf = copy.deepcopy(G)

        # call FF on residual graph
        src = len(Gf) - 1
        sin = len(Gf) - 2
        Gf = self.findMinCut(Gf=Gf, source=src, sink=sin)

        # Call DFS on residual graph to check connectivity
        # classify nodes as FG or BG by checking
        # if each node is connected to source (FG), if not, they are in BG
        # return N*N*1 array, last dim is class 0 (BG) or 1 (FG)
        visited = [False] * len(Gf)
        self.dfs(Gf=Gf, s=src, visited=visited)
        outputArray = [[0] * N] * N

        # traverse all pixel nodes
        # if a node is visited by dfs from s and
        # the edge to a neighbor has become zero, then it is in FG
        for i in range(len(Gf) - 2):
            for j in range(len(Gf) - 2):
                print("i: ", i)
                print("j", j)
                print("Gf[i].capacities[j][0]", Gf[i].capacities[j][0])
                print("Gf[i].capacities[j][0] == 0", Gf[i].capacities[j][0] == 0)
                print("G[i].capacities[j][0] > 0", G[i].capacities[j][0] > 0)
                print("visited[i]", visited[i])
                if j in Gf[i].capacities:
                    if (
                        Gf[i].capacities[j][0] == 0
                        and G[i].capacities[j][0] > 0
                        and visited[i]
                    ):
                        outputArray[G[i].pos[0]][G[i].pos[1]] = 1

        return np.array(outputArray)

    # Ford Fulkerson (FF) Alg for Min Cut
    # remove edges from graph to produce min cut
    def findMinCut(self, Gf, source, sink):

        # All flows start at zero
        maxFlow = 0

        # Use BFS to add flow while there is an s-t path
        while self.bfs(Gf, source, sink):

            # Find max flow through the augmenting path
            # This is equal to the min residual capacity of the
            # edges along the path
            # start augmenting path flow at infinity to make
            # sure any existing capacity is less than it
            augmenting_flow = float("Inf")
            s = sink  # start from sink
            while s != source:
                # update aug flow to min of edge from parent to s and aug flow
                parent = Gf[s].parent
                augmenting_flow = min(augmenting_flow, Gf[parent].capacities[s][0])
                print("poss augmenting flow: ", Gf[parent].capacities[s][0])

                # move to next node in path
                s = Gf[parent].key

            # Add augmenting flow to total max flow
            maxFlow += augmenting_flow

            # Update residual capacities of edges along the path
            # reverse edges if needed
            curr = sink
            while curr != source:
                parent = Gf[curr].parent
                prev = Gf[parent].key

                # decrement resid cap from parent to current node
                Gf[prev].capacities[curr][0] -= augmenting_flow
                Gf[curr].capacities[prev][1] -= augmenting_flow

                # increment resid cap from current to parent
                Gf[prev].capacities[curr][1] += augmenting_flow
                Gf[curr].capacities[prev][0] += augmenting_flow

                curr = Gf[curr].parent

        return Gf

    # BFS for FF
    # This makes my FF the Edmonds-Karp variation of the algorithm
    # reference: https://www.geeksforgeeks.org/minimum-cut-in-a-directed-graph/?ref=rp
    # parent is a list to store the s-t path
    def bfs(self, Gf, s, t):
        # list to store visited nodes
        visited = [False] * len(Gf)

        # queue for BFS
        queue = []

        # add source index to queue and mark as visited
        queue.append(s)
        visited[s] = True

        # BFS loop
        while not not queue:
            # dequeue first node in queue
            u = Gf[queue.pop(0)]  # FIXME do I need copy.copy?

            # Get all adjacent vertices of the dequeued vertex u
            # If a adjacent has not been visited, then mark it
            # visited and enqueue it
            adj = u.edgesOut
            for index in list(reversed(adj)):
                if visited[index] == False and u.capacities[index][0] > 0:
                    # next = copy.copy(Gf[index])
                    queue.append(index)
                    visited[index] = True
                    Gf[index].parent = u.key

        return True if visited[t] else False

    # DFS to traverse residual graph and find cuts
    # reference: https://www.geeksforgeeks.org/minimum-cut-in-a-directed-graph/?ref=rp
    def dfs(self, Gf, s, visited):
        visited[s] = True
        # traverse all pixel nodes)
        for node in Gf[:-2]:
            if Gf[s].capacities[node.key][0] > 0 and not visited[node.key]:
                self.dfs(Gf, node.key, visited)

    # The probability a pixel node is in the foreground
    def foregroundProb(self, x, G):
        N = np.sqrt(len(G) - 2)

        # Get index of x_a in G from x_a coordinates
        x_a_key = int((N * self.x_a[0]) + self.x_a[1])

        print("G[x_a_key].rgb: ", G[x_a_key].rgb)
        print("G[x].rgb", G[x].rgb)

        # Calculate 442 - Euclidean dist between RBG vals of x and x_a
        d = np.round(np.sqrt(((G[x_a_key].rgb - G[x].rgb) ** 2).sum()))
        return int(442 - d)

    # The probability a pixel node is in the background
    def backgroundProb(self, x, G):
        N = np.sqrt(len(G) - 2)

        # Get index of x_b in G from x_b coordinates
        x_b_key = int((N * self.x_b[0]) + self.x_b[1])

        print("G[x_b_key].rgb: ", G[x_b_key].rgb)
        print("G[x].rgb", G[x].rgb)

        # Calculate 442 - Euclidean dist between RBG vals of x and x_b
        d = np.round(np.sqrt(((G[x_b_key].rgb - G[x].rgb) ** 2).sum()))
        return int(442 - d)

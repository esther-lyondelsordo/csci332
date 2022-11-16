"""
CSCI 332
Esther Lyon Delsordo
Software Assignment #1
Due November 16, 2022

Implement Kruskal's algorithm to find the MST of a weighted graph
"""
import numpy as np


class kruskalClass:
    # initialize class values FIXME
    def __init__(self, graph):
        self.graph = graph

    # Function to return a class instance FIXME
    def kruskalClass(self):
        return kruskalClass()

    # find MST from an adjacency matrix TODO
    def findMinimumSpanningTree(A):
        # if A_ij > 0 there is an edge from node i to node j
        # returns MST T, also in adjacency matrix form
        return -1

    # merge method for mergesort
    def merge(left, right):
        arr = np.array([])

        # append the smallest element from each array until one is empty
        while len(left) != 0 and len(right) != 0:
            if left[0] > right[0]:
                arr = np.append(arr, right[0])
                right = np.delete(right, 0)
            else:
                arr = np.append(arr, left[0])
                left = np.delete(left, 0)

        # If right is empty, we need to handle rest of left
        while len(left) != 0:
            arr = np.append(arr, left[0])
            left = np.delete(left, 0)

        # If left is empty, we need to handle rest of right
        while len(right) != 0:
            arr = np.append(arr, right[0])
            right = np.delete(right, 0)

        return arr

    # recursive mergesort algorithm
    def mergesort(a):
        # Sort the values in array ‘a’ and return the sorted array ‘b’.
        n = len(a)

        # base case
        if n == 1:
            return a

        # define some values
        lo = 0
        hi = n - 1
        mid = lo + hi // 2

        # build subarrays
        left = a[lo : mid + 1]
        right = a[mid + 1 : hi + 1]

        # sort subarrays recursively
        left = kruskalClass.mergesort(left)
        right = kruskalClass.mergesort(right)

        return kruskalClass.merge(left, right)

    """
    build the union-find data structure
    input N is the number of nodes in the graph
    return a dictionary
    key = number labels a connected component
    value = array of pointers to nodes in that connected component
    """

    def makeUnionFind(N):
        # Make one big array of all of the nodes, named 1 to N
        arrayOfNodes = np.array(range(1, N + 1))

        # Split the array into an array of arrays
        newArray = np.split(arrayOfNodes, N)

        # use the new array to make a dictionary,
        # each key is the same number as the single array entry value
        u = dict(list(enumerate(newArray, start=1)))

        # If two nodes are connected, assign the parent to the representative node
        return u

    # union function TODO
    # Combine the connected components given by sets named s1 and s2 from u_in
    def union(u_in, s1, s2):
        # find the which set is smaller, rename sets for DRY code
        if len(u_in[s1]) <= len(u_in[s2]):
            small = s1
            big = s2
        else:
            small = s2
            big = s1

        # copy elements of smaller set into larger set
        u_in[big] = np.append(u_in[big], u_in[small][1:])

        # change root of all elements in smaller set to root of larger set
        for i in range(len(u_in[small])):
            temp = u_in[small][i]
            u_in[temp].key = u_in[big].key

        # change name of smaller to name of larger set
        u_in[small].key = u_in[big].key

        # return the updated union-find data structure
        u_out = u_in
        return u_out

    # find function TODO
    # u is the union find data structure and v is the index of a graph node
    # return the label of the set that v belongs to, s
    def find(u, v):
        s = -1
        # the key of the desired index is the name of the root of that set
        s = u[v].key
        return s

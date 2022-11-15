"""
CSCI 332
Esther Lyon Delsordo
Software Assignment #1
Due November 14, 2022

Implement Kruskal's algorithm to find the MST of a weighted graph
"""
import math
import numpy as np


class kruskalClass:
    # initialize class values
    def __init__(self, graph, mst):
        self.name = graph
        self.age = mst

    # find MST from an adjacency matrix
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

    # find function
    def find(u):
        # Return the connected component of the element ‘u’ in ‘S’
        return -1

    # union function
    def union(A, B):
        # Combine the connected components given by node sets ‘A’ and ‘B’ into set C
        return -1

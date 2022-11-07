"""
CSCI 332
Esther Lyon Delsordo
Software Assignment #1
Due November 14, 2022

Implement Kruskal's algorithm to find the MST of a weighted graph
"""
import math
import numpy as np


class KruskalMST:
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
        left = KruskalMST.mergesort(left)
        right = KruskalMST.mergesort(right)

        return KruskalMST.merge(left, right)

    # build union-find data structure
    def makeUnionFind(S):
        # Convert a set ‘S’ into a union-find data structure F
        return -1

    # find function
    def find(u):
        # Return the connected component of the element ‘u’ in ‘S’
        return -1

    # union function
    def union(A, B):
        # Combine the connected components given by node sets ‘A’ and ‘B’ into set C
        return -1

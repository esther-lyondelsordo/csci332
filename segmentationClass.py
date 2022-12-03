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
    def segmentationClass(self):
        return segmentationClass()

    """
    Input: I is an NxNx3 numpy array representing a color (RGB) image. Each pixel
    intensity will be have an integer value between 0 and 255.
    ● Output: L is an NxNx1 numpy array of binary values representing whether each pixel
    in the image is in the foreground (value of 1) or in the background (value of 0). So, if
    pixel at row (𝑖,𝑗) is in the foreground then 𝐿[𝑖,𝑗] = 1, and if it is in the background we
    have 𝐿[𝑖,𝑗] = 0.
    """

    def segmentImage(Img):
        return -1

    # Ford Fulkerson (FF) Alg for Max Flow
    def findMaxFlow(G):
        return -1

    # make residual graph for FF
    def makeResidualGraph(G):
        return -1

    # augment a path in the residual graph for FF
    def augment(path, flow):
        return -1

    # bottleneck function for augment function
    # finds the minimum residual capacity on any edge in an s-t path
    def bottleneck(path, flow):
        return -1

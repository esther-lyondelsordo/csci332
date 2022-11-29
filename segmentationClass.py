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

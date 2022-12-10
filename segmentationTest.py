"""
CSCI 332
Esther Lyon Delsordo
Software Assignment #2
Due December 10, 2022

Implement Ford-Fulkerson Alg and use it to implement the image
segmentation algorithm from Kleinberg-Tardos 7.10

Test script for segmentationClass.py
"""

# Import numpy and pyplot
import numpy as np
import matplotlib.pyplot as plt

# Import my image segmentation class
import segmentationClass

# load a test image FIXME
# I = plt.imread("Louisfjellet25pixel.jpg")

# build test image FIXME
# 3 x 3 with two red pixels, others valued zero
I = np.zeros([3, 3, 3])
I[2, 2, 0] = 128
I[1, 2, 0] = 128

# Make an image segmentation object
# set segmentation object params inline FIXME
sc = segmentationClass.segmentationClass(
    p0=1,  # neigbor penalty
    x_a=np.array([2, 2]),  # Foreground pixel position
    x_b=np.array([0, 0]),  # Background pixel
)

# segment the image
newI = sc.segmentImage(I)

# Plot results
fig, axs = plt.subplots(1, 2)
fig.suptitle("Input and segmentation")
axs[0].imshow(I.astype(np.uint8), interpolation="nearest")
axs[0].set_title("Input image")

# scale 0s and 1s in output image to 0 or 255
axs[1].imshow(255 * newI.astype(np.uint8), interpolation="nearest")
axs[1].set_title("Binary segmentation")
plt.show()

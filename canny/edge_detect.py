import skimage
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters, feature, morphology, measure, segmentation, img_as_ubyte

# Read image
img = io.imread('g.png')

# Convert to grayscale
img_gray = color.rgb2gray(img)

# Apply Canny edge detection
edges = feature.canny(img_gray, sigma=3)

plt.imshow(edges, cmap='gray')

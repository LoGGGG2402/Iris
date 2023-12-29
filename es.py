import math

import cv2
import numpy as np
import scipy

# Load the image
img = cv2.imread("img.png")

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2d image to 1d image
gray = gray.flatten()

# gabor convolution
ksize = 31
sigma = 5
theta = 0
lamda = 10
gamma = 0.5
phi = 0
kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, phi)
kernel = kernel.flatten()
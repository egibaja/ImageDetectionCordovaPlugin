#!/usr/bin/python

# Standard imports
import cv2
import numpy as np;
from matplotlib import pyplot as plt

# Read image
im = cv2.imread("bicho.jpg")

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()


# Filter by Inertia

params.filterByArea = True
minArea  = 20

# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)


# Detect blobs.
keypoints = detector.detect(im)

print(keypoints)
# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
# the size of the circle corresponds to the size of blob

im_with_keypoints = cv2.drawKeypoints(im, keypoints, None, color=(255,0,0), flags=4)


plt.imshow(im_with_keypoints),plt.show()

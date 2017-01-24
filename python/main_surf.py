from __future__ import division
import numpy as np
import cv2
from matplotlib import pyplot as plt

scene = cv2.imread('bicho3.jpg',0)
template = cv2.imread('t2c.png',0)

"""
h, w = template.shape
height, width = scene.shape

r = (w * 1.2) / (width * 1.0)
dim = (int(w * 1.2), int(height * r))

scene = cv2.resize(scene, dim)
"""

surf = cv2.xfeatures2d.SURF_create(1000)
kp1, des1 = surf.detectAndCompute(template,None)

surf2 = cv2.xfeatures2d.SURF_create(1000)  
kp2, des2 = surf.detectAndCompute(scene,None)

print len(kp2)

img4 = cv2.drawKeypoints(template, kp1, None, color=(255,0,0), flags=0)
#plt.imshow(img4),plt.show()
img5 = cv2.drawKeypoints(scene, kp2, None, color=(255,0,0), flags=0)
#plt.imshow(img5),plt.show()

# create BFMatcher object
bf = cv2.BFMatcher(crossCheck=True)
# Match descriptors.
matches = bf.match(des2, des1)

num_matches = len(matches)
max_distance = 0.06
if num_matches < 2:
 	print("######## NO ENCONTRADO")
for m in matches:
	if (m.distance > max_distance):
		print("######## NO ENCONTRADO")

print("--- ENCONTRADO ---")


# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)
for m in matches[:20]:
	print(m.distance)

# Draw first 10 matches.
img3 = cv2.drawMatches(scene,kp2, template,kp1,matches, None, flags=2)

plt.imshow(img3),plt.show()
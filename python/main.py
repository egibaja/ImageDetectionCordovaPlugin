import numpy as np
import cv2
from matplotlib import pyplot as plt

img2 = cv2.imread('bicho2.jpg',0)
img1 = cv2.imread('prim1.png',0)


#ret,th1 = cv2.threshold(img1,100,255,cv2.THRESH_BINARY)
#ret,th2 = cv2.threshold(img2,100,255,cv2.THRESH_BINARY)

# Initiate STAR detector
# int nfeatures=500, float scaleFactor=1.2f, int nlevels=8, int edgeThreshold=31, int firstLevel=0, int WTA_K=2, int scoreType=ORB::HARRIS_SCORE, int patchSize=31, int fastThreshold=20
orb = cv2.ORB_create(nfeatures=100, edgeThreshold=15, patchSize=15)
orb2 = cv2.ORB_create(nfeatures=100, edgeThreshold=21, patchSize=21)

# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb2.detectAndCompute(img2,None)
# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des2, des1)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

for m in matches[:20]:
	print(m.distance)

# Draw first 10 matches.
img3 = cv2.drawMatches(img1,kp1, img2,kp2,matches[:20], None, flags=2)


img4 = cv2.drawKeypoints(img1, kp1, None, color=(255,0,0), flags=0)
plt.imshow(img4),plt.show()
img5 = cv2.drawKeypoints(img2, kp2, None, color=(255,0,0), flags=0)
plt.imshow(img5),plt.show()

plt.imshow(img3),plt.show()



#http://stackoverflow.com/questions/31630559/attributeerror-module-object-has-no-attribute-orb
#http://stackoverflow.com/questions/31631352/typeerror-required-argument-outimg-pos-6-not-found/31631995#31631995
# probar cosas del email de Nano
# probar codigo en plugin de Image clouduki whatever


# Este parece muy bueno.
# http://stackoverflow.com/questions/10666436/scale-and-rotation-template-matching
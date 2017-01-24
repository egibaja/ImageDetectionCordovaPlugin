import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread('c1.png',0)
img2 = cv2.imread('bicho.jpg',0)

# Initiate STAR detector
# int nfeatures=500, float scaleFactor=1.2f, int nlevels=8, int edgeThreshold=31, int firstLevel=0, int WTA_K=2, int scoreType=ORB::HARRIS_SCORE, int patchSize=31, int fastThreshold=20
orb = cv2.ORB_create(nfeatures=500, edgeThreshold=50, patchSize=50)

# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)
# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

for m in matches[:20]:
	print(m.distance)

# Draw first 10 matches.
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10], None, flags=2)

plt.imshow(img3),plt.show()



#http://stackoverflow.com/questions/31630559/attributeerror-module-object-has-no-attribute-orb
#http://stackoverflow.com/questions/31631352/typeerror-required-argument-outimg-pos-6-not-found/31631995#31631995
# probar cosas del email de Nano
# probar codigo en plugin de Image clouduki whatever


"""

# find the keypoints with ORB
kp = orb.detect(img,None)

# compute the descriptors with ORB
kp, des = orb.compute(img, kp)

# draw only keypoints location,not size and orientation
img2 = cv2.drawKeypoints(img,kp,None,color=(255,0,0), flags=0)
plt.imshow(img2),plt.show()
"""
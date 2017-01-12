import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('t2',0)

# Initiate STAR detector
# int nfeatures=500, float scaleFactor=1.2f, int nlevels=8, int edgeThreshold=31, int firstLevel=0, int WTA_K=2, int scoreType=ORB::HARRIS_SCORE, int patchSize=31, int fastThreshold=20
orb = cv2.ORB_create(nfeatures=10000, edgeThreshold=3)

# find the keypoints with ORB
kp = orb.detect(img,None)

# compute the descriptors with ORB
kp, des = orb.compute(img, kp)

# draw only keypoints location,not size and orientation
img2 = cv2.drawKeypoints(img, kp, None color=(255,0,0), flags=0)
plt.imshow(img2),plt.show()


#http://stackoverflow.com/questions/31630559/attributeerror-module-object-has-no-attribute-orb
#http://stackoverflow.com/questions/31631352/typeerror-required-argument-outimg-pos-6-not-found/31631995#31631995
# probar cosas del email de Nano
# probar codigo en plugin de Image clouduki whatever
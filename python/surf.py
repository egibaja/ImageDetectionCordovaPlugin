import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('t2c.png', 0)

surf = cv2.xfeatures2d.SURF_create(1000)

kp, des = surf.detectAndCompute(img,None)

print len(kp)

# draw only keypoints location,not size and orientation
img2 = cv2.drawKeypoints(img,kp,None,(255,0,0),4)
plt.imshow(img2),plt.show()


#http://stackoverflow.com/questions/31630559/attributeerror-module-object-has-no-attribute-orb
#http://stackoverflow.com/questions/31631352/typeerror-required-argument-outimg-pos-6-not-found/31631995#31631995
# probar cosas del email de Nano
# probar codigo en plugin de Image clouduki whatever
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('bicho2.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

lower_red = np.array([0,50,50])
upper_red = np.array([10,255,255])
mask0 = cv2.inRange(hsv, lower_red, upper_red)

lower_red = np.array([170,50,50])
upper_red = np.array([180,255,255])
mask1 = cv2.inRange(hsv, lower_red, upper_red)

mask = mask0+mask1

res = cv2.bitwise_not(img,img, mask= mask)


plt.imshow(img),plt.show()
plt.imshow(mask),plt.show()
plt.imshow(res),plt.show()

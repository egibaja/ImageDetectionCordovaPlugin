import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('bicho2.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_red = np.array([0,30,50])
upper_red = np.array([20,255,255])
mask0 = cv2.inRange(hsv, lower_red, upper_red)

res = cv2.bitwise_not(img,img, mask= mask0)


plt.imshow(img),plt.show()
plt.imshow(mask0),plt.show()
plt.imshow(res),plt.show()

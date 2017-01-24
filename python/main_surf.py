from __future__ import division
import numpy as np
import cv2
from matplotlib import pyplot as plt
import glob, os


HESSIAN_LOGO = 1000
HESSIAN_LOGO_B = 10000
HESSIAN_SCENE = 1000
HESSIAN_SCENE_B = 1000


LOGO_SRC = '../../comb.png'
LOGO_B_SRC = '../../comb_b.png'
SCENE_SRC = 'bicho.jpg'


def drawKp(image, kp):
    img = cv2.drawKeypoints(image, kp, None, color=(255,0,0), flags=0)
    plt.imshow(img),plt.show()

def drawMatches(image1, kp1, image2, kp2, matches):
    img = cv2.drawMatches(image1, kp1, image2, kp2, matches, None, flags=2) 
    plt.imshow(img),plt.show()

def resize(template, scene):
    #resize scene image
    h, w = template.shape
    height, width = scene.shape

    r = (w * 1.2) / (width * 1.0)
    dim = (int(w * 1.2), int(height * r))

    return cv2.resize(scene, dim)

def detectAndCompute(image, hessian_threshold):
    surf = cv2.xfeatures2d.SURF_create(hessian_threshold)
    return surf.detectAndCompute(image,None)

def getMatches(des1, des2):
    # create BFMatcher object
    bf = cv2.BFMatcher(crossCheck=True)
    # Match descriptors.
    return bf.match(des1, des2)

def isLogoPresent(matches):
	# check if the logo is present
    is_logo_match = True
    max_distance_logo = 0.08
    if matches[0].distance > max_distance_logo:
    	is_logo_match = False
    if matches[-1].distance > (matches[0].distance * 2.5):
    	is_logo_match = False

    #print("-- Max dist in first ten points: ", matches[-1].distance ) 
    #print("-- Min dist in first ten points: ", matches[0].distance ) 
     
    #for m in matches: 
    #  print(m.distance)

    return is_logo_match

def isBottomPresent(matches):
	 # check if we found the two rectangles
    max_distance = 0.06
    is_bottom_match = True
    if len(matches) < 2:
        is_bottom_match = False
    else:
        matches = sorted(matches, key = lambda x:x.distance)
        for m in matches[:1]:
            if (m.distance > max_distance):
        		is_bottom_match = False

    #for m in matches: 
    #  print(m.distance)

    return is_bottom_match


def main(src):
    logo = cv2.imread(LOGO_SRC,0)
    logo_bottom = cv2.imread(LOGO_B_SRC,0)
    scene = cv2.imread(src,0)
    scene = resize(logo, scene)

    h,w = scene.shape
    scene_top = scene[0: int(h * 0.3) , 0:w]
    scene_bottom = scene[int(h * 0.7): h , 0:w]

    kp_logo, des_logo = detectAndCompute(logo, HESSIAN_LOGO)
    #print("--- Logo: hessian value %i, number of descriptors %i" %(HESSIAN_LOGO, len(kp_logo)))
    kp_logo_bottom, des_logo_bottom = detectAndCompute(logo_bottom, HESSIAN_LOGO_B)
    #print("--- Bottom: hessian value %i, number of descriptors %i" %(HESSIAN_LOGO_B, len(kp_logo_bottom)))
    kp_scene_top, des_scene_top = detectAndCompute(scene_top, HESSIAN_SCENE)
    #print("--- Scene_top: hessian value %i, number of descriptors %i" %(HESSIAN_SCENE, len(kp_scene_top)))
    kp_scene_bottom, des_scene_bottom = detectAndCompute(scene_bottom, HESSIAN_SCENE_B)
    #print("--- Scene_bottom: hessian value %i, number of descriptors %i" %(HESSIAN_SCENE_B, len(kp_scene_bottom)))

    matches_bottom = getMatches(des_scene_bottom, des_logo_bottom)
    matches_logo = getMatches(des_scene_top, des_logo)
    #print("%i matches between scene_top and logo" %(len(matches_logo)))
    #print("%i matches between scene_bottom and bottom" %(len(matches_bottom)))


    # check if we found the two rectangles
    is_bottom_match = isBottomPresent(matches_bottom)
 
    # check if the logo is present
    matches_logo_aux = sorted(matches_logo, key = lambda x:x.distance)[:10]
    is_logo_match = isLogoPresent(matches_logo_aux)

    if is_logo_match and is_bottom_match:
    	print("ENCONTRADO  (%s y %s)" %(is_logo_match, is_bottom_match))
    else:
    	print("---> fallo  (%s y %s)" %(is_logo_match, is_bottom_match))
    


    #drawKp(scene_top, kp_scene_top)
    #drawKp(scene_bottom, kp_scene_bottom)
    #drawKp(logo, kp_logo)
    #drawKp(logo_bottom, kp_logo_bottom)
    #drawMatches(scene_bottom, kp_scene_bottom, logo_bottom, kp_logo_bottom, matches_bottom)
    #drawMatches(scene_top,kp_scene_top, logo,kp_logo,matches_logo_aux)


if __name__ == "__main__":
	os.chdir("./images/frominstance/")
	for file in glob.glob("*.jpeg"):
		print(" %s ......................" %(file))
		main(file)

from __future__ import division
import numpy as np
import cv2
from matplotlib import pyplot as plt
import glob, os


LOGO_SRC = 'comb_a.png'
LOGO_B_SRC = 'comb_b.png'
SCENE_SRC = 'bicho2.jpg'

SETTINGS_LOGO = {'nfeatures': 1000, 'edgeThreshold' : 8}
SETTINGS_LOGO_B = {'nfeatures': 500, 'edgeThreshold' : 40}
SETTINGS_SCENE = {'nfeatures': 500, 'edgeThreshold' : 35}
SETTINGS_SCENE_B = {'nfeatures': 1000, 'edgeThreshold' : 20}


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

def detectAndCompute(image, settings):
    orb = cv2.ORB_create(**settings)
    return orb.detectAndCompute(image,None)

def getMatches(des1, des2):
    # create BFMatcher object
    bf = cv2.BFMatcher()
    # Match descriptors.
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)
    return good

def isLogoPresent(matches):
	# check if the logo is present
    is_logo_match = True
    max_distance_logo = 0.08
    if matches[0].distance > max_distance_logo:
    	is_logo_match = False
    if matches[-1].distance > (matches[0].distance * 2.5):
    	is_logo_match = False

    print("-- Max dist in first ten points: ", matches[-1].distance ) 
    print("-- Min dist in first ten points: ", matches[0].distance ) 
     
    #for m in matches: 
    #  print(m.distance)

    return is_logo_match

def isBottomPresent(matches):
	 # check if we found the two rectangles
    max_distance = 33.0
    is_bottom_match = True
    if len(matches) < 4:
        is_bottom_match = False
    else:
        matches = sorted(matches, key = lambda x:x.distance)
        for m in matches[:4]:
            if (m.distance > max_distance):
        		is_bottom_match = False
        		break

    #for m in matches[:4]: 
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

    kp_logo, des_logo = detectAndCompute(logo, SETTINGS_LOGO)
    print("--- Logo: number of descriptors %i" %(len(kp_logo)))
    kp_logo_bottom, des_logo_bottom = detectAndCompute(logo_bottom, SETTINGS_LOGO_B)
    print("--- Bottom: number of descriptors %i" %(len(kp_logo_bottom)))
    kp_scene_top, des_scene_top = detectAndCompute(scene_top, SETTINGS_SCENE)
    print("--- Scene_top: number of descriptors %i" %(len(kp_scene_top)))
    kp_scene_bottom, des_scene_bottom = detectAndCompute(scene_bottom, SETTINGS_SCENE_B)
    print("--- Scene_bottom: number of descriptors %i" %(len(kp_scene_bottom)))

    matches_bottom = getMatches(des_scene_bottom, des_logo_bottom)
   
    matches_logo = getMatches(des_scene_top, des_logo)

    print("%i matches between scene_top and logo" %(len(matches_logo)))
    print("%i matches between scene_bottom and bottom" %(len(matches_bottom)))

    # check if we found the two rectangles
    #is_bottom_match = isBottomPresent(matches_bottom)
    is_bottom_match = len(matches_bottom) > 1
 
    # check if the logo is present
    #matches_logo_aux = sorted(matches_logo, key = lambda x:x.distance)[:10]
    #is_logo_match = isLogoPresent(matches_logo_aux)
    is_logo_match = len(matches_logo) > 10

    if is_logo_match and is_bottom_match:
    	print("hay logo y puntos")
        return True
    else:
    	print("---not found: logo y puntos = (%s y %s)" %(is_logo_match, is_bottom_match))
        return False
    


    drawKp(scene_top, kp_scene_top)
    drawKp(scene_bottom, kp_scene_bottom)
    #drawKp(logo, kp_logo)
    #drawKp(logo_bottom, kp_logo_bottom)
    drawMatches(scene_bottom, kp_scene_bottom, logo_bottom, kp_logo_bottom, matches_bottom[:4])
    drawMatches(scene_top,kp_scene_top, logo,kp_logo,matches_logo)


if __name__ == "__main__":
    os.chdir(".")
    num_valids = 0
    num_invalids = 0
    for file in glob.glob("./images/valids/*.jpeg"):
    	print(" %s ......................" %(file))
    	if main(file):
            num_valids += 1
        else:
            num_invalids += 1
    print("____________________________________________________")
    print("%i Imagenes VALIDAS" %(num_valids))
    print("%i Imagenes NO VALIDAS" %(num_invalids))

	#main(SCENE_SRC)

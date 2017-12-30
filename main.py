# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 11:59:05 2017

@author: Antonio Serrano
"""


import numpy as np
import cv2
from matplotlib import pyplot as plt
import warnings


def show_image (img) :
     
     """It shows the image given (black and white)
     """    
     plt.imshow(img, cmap='Greys_r')   
     plt.xticks([]), plt.yticks([])
     plt.show() 
    
     warnings.simplefilter("ignore") 
     plt.pause(1)
     print ("Press a key to continue...\n")
     plt.waitforbuttonpress(0)
     plt.close()  
     

def get_kp_and_desc (img):
     
     """ It gets the keypoints and descriptors of the image given
     """
     # Creation of SIFT object   
     sift = cv2.xfeatures2d.SIFT_create()
     
     # Getting keypoints and descriptors
     kp, descriptor = sift.detectAndCompute(img, None)
          
     return kp, descriptor
     
def match_points(img1, img2, n_points, method) :
     
     """ It matches n_points given of the two images, according to the 
     method given (brute force or knnMatch)
     
     Reference: https://docs.opencv.org/3.3.0/dc/dc3/tutorial_py_matcher.html
     """
     
     # Getting keypoints and descriptors from both images
     kp1, descriptor1 = get_kp_and_desc(img1)
     kp2, descriptor2 = get_kp_and_desc(img2)
     
     # Using bruteForce
     if (method=="bruteForce") :
          
          # Creating BFMatcher object
          my_bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
     
          # Getting matches between both descriptors
          matches = my_bf.match(descriptor1,descriptor2)
          
          # Sorting matches according to the distance
          matches = sorted(matches, key = lambda x:x.distance)
     
          # Drawing N matches
          result = cv2.drawMatches(img1, kp1, img2, kp2, matches[:n_points], 
                                   flags=2, outImg=None)
     
     # Using knnMatch with k=2
     else :
          
          # Same as before
          my_bf = cv2.BFMatcher()
          matches = my_bf.knnMatch(descriptor1,descriptor2, k=2)
        
          # Taking match only if the distance's proportion is met
          best_one = []
          for x,y in matches:
              if (x.distance < 0.7*y.distance):
                  best_one.append([x])
         
          # Drawing N matches
          result = cv2.drawMatchesKnn(img1, kp1, img2, kp2, 
                                      best_one[:n_points],flags=2, outImg=None)
     
     # Showing result
     show_image(result)
          

def get_homography (img1,img2):
     
     """It calculates homography between two images
     """
     
     # Getting keypoints and descriptors from both images
     kp1, descriptor1 = get_kp_and_desc(img1)
     kp2, descriptor2 = get_kp_and_desc(img2)
     
     # Creating BFMatcher object
     my_bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
     
     # Getting matches bewteen descriptors
     matches = my_bf.match(descriptor1,descriptor2)
          
     # Sorting matches according to the distance
     matches = sorted(matches, key = lambda x:x.distance)
     
     # Getting keypoints from the best matches
     keypoints1 = np.float32([kp1[i.queryIdx].pt for i in matches])
     keypoints2 = np.float32([kp2[i.trainIdx].pt for i in matches])
     
     # Getting homography H
     H, mask = cv2.findHomography(srcPoints=keypoints1,dstPoints=keypoints2,
                                  method=cv2.RANSAC,ransacReprojThreshold=1)     
          
     return H
     
def size_mosaic(list_images):
     
     """It calculates the size of the mosaic, given a list of images
     """
     weidth = 0
     height = 0
     
     for image in list_images:
          weidth += image.shape[1]
          height += image.shape[0]
                    
     return height,weidth

def crop_mosaic(mosaic) :

     """It crops the useless part of the mosaic
     
     References: 
      https://docs.opencv.org/trunk/d9/d8b/tutorial_py_contours_hierarchy.html
      https://docs.opencv.org/3.3.0/d4/d73/tutorial_py_contours_begin.html
      https://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html
     
     """  
     # Getting binary image
     retval,binary = cv2.threshold(mosaic,1,255,cv2.THRESH_BINARY)

     # Getting edges of the image
     image,contours,hierarchy = cv2.findContours(image=binary,
                        mode=cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_SIMPLE)
     cnt = contours[0]
     
     # Getting rectangle that wraps the edges of the image
     x,y,w,h = cv2.boundingRect(cnt)
     
     # Cropping mosaic
     crop = mosaic[y:y+h, x:x+w]

     return crop

def make_mosaic(list_images) : 

     """It creates a panoramic image with the list of images given
     """
     
     # 1. Creating canvas (mosaic)
     height, weidth  = size_mosaic(list_images)
     mosaic = (np.ones((height,weidth),np.uint8)*0)
     
     # 1. Getting index of central image
     central = int(len(list_images)/2)
     
     # Putting central image into the mosaic using the proper homography
     x = mosaic.shape[0]/2 - list_images[central].shape[0]/2
     y = mosaic.shape[1]/2 - list_images[central].shape[1]/2
     traslation = np.float32([[1,0,y],[0,1,x],[0,0,1]])
                    
     cv2.warpPerspective(src=list_images[central], M=traslation, dst=mosaic, 
     dsize=(mosaic.shape[1],mosaic.shape[0]),borderMode=cv2.BORDER_TRANSPARENT)

     # 3. Getting left and right sides homographies 
     left = traslation.copy()
     right = traslation.copy()
  
     # Putting all the images into the mosaic
     
     # From the center to the left side
     for i in range(central, 0, -1):
                                 
          H = get_homography(list_images[i-1],list_images[i])
          H = np.dot(left, H)
          cv2.warpPerspective(src=list_images[i-1], dst=mosaic, M=H, 
    dsize=(mosaic.shape[1], mosaic.shape[0]),borderMode=cv2.BORDER_TRANSPARENT)

          left = H

     # From the center to the right side
     for i in range(central, len(list_images)-1):

          H = get_homography(list_images[i+1],list_images[i])
          H = np.dot(right, H)
          cv2.warpPerspective(src=list_images[i+1], dst=mosaic, M=H, 
    dsize=(mosaic.shape[1], mosaic.shape[0]),borderMode=cv2.BORDER_TRANSPARENT)
          
          right = H
          
     # 4. Cropping the useless part (black parts) of the mosaic
     mosaic = crop_mosaic(mosaic)     
     
     show_image(mosaic)
     
    
# =============================================================================
# MAIN  
# =============================================================================
if __name__ == '__main__':
    
    
    # Reading images
    path1 = "images/Yosemite1.jpg"
    path2 = "images/Yosemite2.jpg"
    img1 = cv2.imread(path1,0)
    img2 = cv2.imread(path2,0)
    
    # Images for the mosaic
    path3 = "images/mosaico002.jpg"
    path4 = "images/mosaico003.jpg"
    path5 = "images/mosaico004.jpg"
    path6 = "images/mosaico005.jpg"
    path7 = "images/mosaico006.jpg"
    path8 = "images/mosaico007.jpg"
    path9 = "images/mosaico008.jpg"
    path10 = "images/mosaico009.jpg"
    path11 = "images/mosaico010.jpg"
    path12 = "images/mosaico011.jpg"
    
    img3 = cv2.imread(path3,0)
    img4 = cv2.imread(path4,0)
    img5 = cv2.imread(path5,0)
    img6 = cv2.imread(path6,0)
    img7 = cv2.imread(path7,0)
    img8 = cv2.imread(path8,0)
    img9 = cv2.imread(path9,0)
    img10 = cv2.imread(path10,0)
    img11 = cv2.imread(path11,0)
    img12 = cv2.imread(path12,0)

    list_images = [img3,img4,img5,img6,img7,img8,img9,img10,img11,img12]

    print ("----------------------------------------------------------------------")
    print ("SIFT descriptor and matching")
    print ("----------------------------------------------------------------------")
    
    print ("Using brueForce")
    match_points(img1,img2,n_points=200, method="bruteForce")
    print ("Using knnMatch")
    match_points(img1,img2,n_points=200, method="knnMatch")
    
    print ("----------------------------------------------------------------------")
    print ("Making mosaic")
    print ("----------------------------------------------------------------------")
    make_mosaic(list_images)

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from MatchPoint import MatchPoint

#reading image
img1 = cv2.imread('img1.jpg')
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

img2 = cv2.imread('img2.jpg')
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

#keypoints
sift1 = cv2.SIFT_create()
kp1, des1 = sift1.detectAndCompute(gray1,None) #kp: list of key points, des:numpy array of shape num_keypointsx128
print("length of kp1: ",len(kp1))
print("shape of des1: ", des1.shape)

sift2 = cv2.SIFT_create()
kp2, des2 = sift2.detectAndCompute(gray2, None)
print("length of kp2: ", len(kp2))
print("shape of des2: ", des2.shape)

img_1 = cv2.drawKeypoints(gray1,kp1,img1)
img_2 = cv2.drawKeypoints(gray2, kp2, img2)

#caculate euclidean distance of each descriptor in image1 with respect to each descriptor in image2
#select the least distance that represent correspondence of descriptors
distance_min_list = []
index_des2 = []
for i in range(len(des1)):
    distance_list = []
    for j in range(len(des2)):
        distance_list.append(cv2.norm(des1[i], des2[j]))

    min_val = distance_list[0]
    min_index = 0
    for k in range(1, len(distance_list)):
        if distance_list[k] < min_val:
            min_val = distance_list[k]
            min_index = k
    matchpt = [min_val, i, min_index] #[distance, des1 index, des2 index]
    distance_min_list.append(matchpt)

distance_min_list = sorted(distance_min_list, key = lambda a:a[0])

for m in distance_min_list:
    print("distance: ", m[0])
    print("des1 index: ", m[1])
    print("des2 index: ", m[2])

# bf = cv2.BFMatcher(cv2.NORM_L2) #Brute-Force
# matches = bf.match(des1, des2)
# matches = sorted(matches, key = lambda x:x.distance)
#
# img3 = cv2.drawMatches(gray1, kp1, gray2, kp2, matches[:10], None)
#
# cv2.imshow("Maching result", img3)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows

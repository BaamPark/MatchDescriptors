# MatchDescriptors

This Python script provides functions for finding matching keypoints between two images, marking them on the images, and then using the RANSAC algorithm to calculate the homography between the two images.

## Dependencies
The script requires the following dependencies:
- OpenCV (cv2)
- NumPy

## Functions
### `mark_matching_points(img1, img2, kp1, kp2, distance_min_list, length_kp)`
This function takes in two images (`img1` and `img2`), two lists of keypoints (`kp1` and `kp2`), a list of distances between keypoints (`distance_min_list`), and the length of the keypoints list (`length_kp`). It then marks the matching keypoints on the images using circles and numbers.

### `ransac(kp1_list, kp2_list, thresh)`
This function takes in two lists of keypoints (`kp1_list` and `kp2_list`) and a threshold value (`thresh`). It then uses the RANSAC algorithm to calculate the homography between the two images based on the matching keypoints. The function returns the final homography matrix.

### `geometricDistance(correspondence, h)`
This function takes in a correspondence pair (`correspondence`) and a homography matrix (`h`). It calculates the geometric distance between the estimated point and the actual point and returns the distance.

### `calculateHomography(correspondences)`
This function takes in a list of correspondence pairs (`correspondences`) and calculates the homography matrix using SVD decomposition. It then returns the homography matrix.

## Usage
To use the functions, import the script and call the functions with the required arguments. For example:

```python
import cv2
import numpy as np
from script_name import mark_matching_points, ransac

# Load images
img1 = cv2.imread('image1.jpg')
img2 = cv2.imread('image2.jpg')

# Find keypoints and descriptors
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# Match keypoints
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

# Find matching keypoints and mark on images
distance_min_list = [[match.distance, match.queryIdx, match.trainIdx] for match in matches]
mark_matching_points(img1, img2, kp1, kp2, distance_min_list, len(distance_min_list))

# Use RANSAC to calculate homography matrix
kp1_list = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
kp2_list = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
finalH = ransac(kp1_list, kp2_list, 0.5)

# Warp image 1 to image 2 using homography matrix
result = cv2.warpPerspective(img1, finalH, (img1.shape[1] + img2.shape[1], img1.shape[0]))
result[0:img2.shape[0], 0:img2.shape[1]] = img2

# Display result
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

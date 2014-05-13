#-*-coding:utf-8-*-

import numpy as np
import cv2
import os
import sys

if __name__ == '__main__':
	img1 = cv2.imread(os.path.join('data', 'box.png'),0)          # queryImage
	img2 = cv2.imread(os.path.join('data', 'box_in_scene.png'),0) # trainImage

	# Initiate SIFT detector
	orb = cv2.ORB()

	# find the keypoints and descriptors with SIFT
	kp1, des1 = orb.detectAndCompute(img1,None)
	kp2, des2 = orb.detectAndCompute(img2,None)

	# create BFMatcher object
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

	# Match descriptors.
	matches = bf.match(des1,des2)

	# Sort them in the order of their distance.
	matches = sorted(matches, key = lambda x:x.distance)

#	for match in matches[:10]:
#		print match.distance, match.trainIdx, match.queryIdx, match.imgIdx
#	sys.exit(0)

	# Draw first 10 matches.
	img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:30], flags=2)

	cv2.imshow('hello', img3)
	cv2.waitKey()


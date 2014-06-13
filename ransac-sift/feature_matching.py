#-*-coding:utf-8-*-

import numpy as np
import cv2
import os
import sys
from find_obj import filter_matches,explore_match
import itertools
import getopt

if __name__ == '__main__':
	query_image = None
	train_image = None
	output_image = None

	opts, args = getopt.getopt(sys.argv[1:], 'q:t:o:', ['query=', 'train=', 'output='])
	for o, a in opts:
		if o in ('-q', '--query'):
			query_image = a
		elif o in ('-t', '--train'):
			train_image = a
		elif o in ('-o', '--output'):
			output_image = a
	assert query_image and train_image

	img1 = cv2.imread(query_image,0)          # queryImage
	img2 = cv2.imread(train_image,0) # trainImage

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
	# 取前30个
	matches = sorted(matches, key = lambda x:x.distance)
	good_matches = matches[:30]

	# 显示匹配
	mkp1 = [kp1[m.queryIdx] for m in good_matches]
	mkp2 = [kp2[m.trainIdx] for m in good_matches]
	kp_pairs = zip(mkp1, mkp2)
	explore_match('BFMatcher', img1,img2,kp_pairs) #cv2 shows image


	# 进行Ransac过程
	src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1, 1, 2)
	dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1, 1, 2)
	assert src_pts.shape == dst_pts.shape

	# 这个函数需要再看下
	# M: 变换矩阵 3 * 3
	# mask: 30 * 1维的01矩阵，代表点对的选择或遗弃。1表示选择
	M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

	height, width = img1.shape;
	result = cv2.warpPerspective(img1, M, (width * 2, height * 2))

	# result = cv2.resize(result, (width, height))
	cv2.imshow('wrapped', result)

	# 根据掩码筛选keypoint
	mask = mask.ravel().tolist()
	mkp1 = itertools.compress(mkp1, mask)
	mkp2 = itertools.compress(mkp2, mask)	

	kp_pairs = zip(mkp1, mkp2)
	explore_match('Ransaced', img1,img2,kp_pairs, output_img = output_image) #cv2 shows image

	cv2.waitKey()
	cv2.destroyAllWindows()

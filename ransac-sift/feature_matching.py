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
	query_keypoints_image = None
	train_keypoints_image = None

	original_match_image = None
	output_image = None
	transformed_image = None

	# 从命令行中读取参数
	opts, args = getopt.getopt(sys.argv[1:], \
		'q:t:o:', \
		['query=', 'train=', 'output=', 'original=', 'transformed=', 'query-keypoints=', 'train-keypoints='])
	for o, a in opts:
		if o in ('-q', '--query'):
			query_image = a
		elif o in ('-t', '--train'):
			train_image = a
		elif o in ('-o', '--output'):
			output_image = a
		elif o in ('--original'):
			original_match_image = a
		elif o in ('--transformed'):
			transformed_image = a
		elif o in ('--query-keypoints'):
			query_keypoints_image = a
		elif o in ('--train-keypoints'):
			train_keypoints_image = a
	assert query_image and train_image

	img1 = cv2.imread(query_image,0)          # queryImage
	img2 = cv2.imread(train_image,0) # trainImage

	# Initiate SIFT detector
	sift = cv2.SIFT()
	# find the keypoints and descriptors with SIFT
	kp1, des1 = sift.detectAndCompute(img1,None)
	kp2, des2 = sift.detectAndCompute(img2,None)

	bf = cv2.BFMatcher()
	# query中的每个KeyPoint在train中都有两个对应点
	matches = bf.knnMatch(des1,des2, k = 2)

	# 筛选对应点，最近邻和次近邻的距离大小满足一定关系
	good_matches = []
	for m, n in matches:
		if m.distance < 0.75 * n.distance:
			good_matches.append(m)

	# 输出Keypoints
	if query_keypoints_image:
		img1_with_keypoints = cv2.drawKeypoints(img1, kp1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
		cv2.imwrite(query_keypoints_image, img1_with_keypoints)
	if train_keypoints_image:
		img2_with_keypoints = cv2.drawKeypoints(img2, kp2, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
		cv2.imwrite(train_keypoints_image, img2_with_keypoints)

	# 显示匹配
	mkp1 = [kp1[m.queryIdx] for m in good_matches]
	mkp2 = [kp2[m.trainIdx] for m in good_matches]
	kp_pairs = zip(mkp1, mkp2)
	explore_match('BFMatcher', img1,img2,kp_pairs, output_img = original_match_image) #cv2 shows image


	# 从KeyPoint中提取位置坐标
	src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1, 1, 2)
	dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1, 1, 2)
	assert src_pts.shape == dst_pts.shape

	# 使用RANSAC方法查找Homography变换
	# M: 变换矩阵 3 * 3
	# mask: 30 * 1维的01矩阵，代表点对的选择或遗弃。1表示选择
	M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
	
	print '%d / %d' % (len(filter(lambda x: x > 0, mask.ravel().tolist())), mask.shape[0])
	print M

	height, width = img1.shape;
	result = cv2.warpPerspective(img1, M, (width * 2, height * 2))

	# result = cv2.resize(result, (width, height))
	if transformed_image:
		cv2.imwrite(transformed_image, result)
	else:
		cv2.imshow('warpped', result)

	# 根据掩码筛选keypoint
	mask = mask.ravel().tolist()
	mkp1 = itertools.compress(mkp1, mask)
	mkp2 = itertools.compress(mkp2, mask)	

	kp_pairs = zip(mkp1, mkp2)
	explore_match('Ransaced', img1,img2,kp_pairs, output_img = output_image) #cv2 shows image

	cv2.waitKey()
	cv2.destroyAllWindows()

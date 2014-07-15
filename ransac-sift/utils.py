#-*-coding:utf-8-*-

import cv2
import numpy as np

def construct_kp_pairs(query_pt_set, train_pt_set):
	kp1 = []
	for pt in query_pt_set:
		keyPoint = cv2.KeyPoint()
		keyPoint.pt = tuple(pt)
		kp1.append(keyPoint)
	kp2 = []
	for pt in train_pt_set:
		keyPoint = cv2.KeyPoint()
		keyPoint.pt = tuple(pt)
		kp2.append(keyPoint)
	kp_pairs = zip(kp1, kp2)
	return kp_pairs

def cut_black_edge(image):
	height, width = image.shape
	print width, height

	hist = np.zeros(width)
	for i in xrange(width):
		col = image[:, i]
		hist[i] = np.count_nonzero(col)
	left, right = 0, 0
	for i in xrange(width):
		if hist[i] > 0:
			left = i
			break
	for i in reversed(xrange(width)):
		if hist[i] > 0:
			right = i
			break

	
	hist = np.zeros(height)
	for i in xrange(height):
		row = image[i, :]
		hist[i] = np.count_nonzero(row)
	top, bottom = 0, 0
	for i in xrange(height):
		if hist[i] > 0:
			top = i
			break
	for i in reversed(xrange(height)):
		if hist[i] > 0:
			bottom = i
			break
	
	roi = image[top:bottom, left:right]
	return roi.copy()

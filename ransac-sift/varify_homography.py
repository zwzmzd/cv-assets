#-*-coding:utf-8-*-

import numpy as np
import cv2
import os
import sys
from find_obj import filter_matches,explore_match
import itertools
import getopt
import json

if __name__ == '__main__':
	query_image = 'data/a1.png'
	train_image = 'data/a2.png'
	original_match_image = None
	output_image = None
	pt_match_file = None

	opts, args = getopt.getopt(sys.argv[1:], 'q:t:o:', ['query=', 'train=', 'output=', 'original=', 'pointsmatch='])
	for o, a in opts:
		if o in ('-q', '--query'):
			query_image = a
		elif o in ('-t', '--train'):
			train_image = a
		elif o in ('-o', '--output'):
			output_image = a
		elif o in ('--original'):
			original_match_image = a
		elif o in ('--pointsmatch'):
			pt_match_file = a
	assert query_image and train_image

	img1 = cv2.imread(query_image,0)          # queryImage
	img2 = cv2.imread(train_image,0) # trainImage

	if pt_match_file:
		# 从文件中读取标注好的点
		fp = open(pt_match_file, 'r')
		query_pt_set, train_pt_set = json.load(fp)
		fp.close()

		src_pts = np.float32(query_pt_set).reshape(-1, 1, 2)
		dst_pts = np.float32(train_pt_set).reshape(-1, 1, 2)

		# 使用8个对应点构造Homography
		M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
		print 'mask:', mask.ravel()

		height, width = img1.shape;
		result = cv2.warpPerspective(img1, M, (width * 2, height * 2))
		cv2.imshow('result', result);

		cv2.waitKey()
		cv2.destroyAllWindows()
	else:
		# 标注对应的点, 并保存到文件中
		img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
		img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

		def mouseWrapper(win, image, collector = None):
			cancel = False
			def onmouse(event, x, y, flags, param):
				global cancel
				if event == cv2.EVENT_LBUTTONDOWN:
					cancel = False
				elif event == cv2.EVENT_MOUSEMOVE:
					cancel = True
				elif event == cv2.EVENT_LBUTTONUP:
					if not cancel:
						collector.append((x, y))
						cv2.circle(image, (x, y), 3, (0, 255, 0), -1, lineType = cv2.CV_AA)
						cv2.imshow(win, image)
			return onmouse

		query_win = 'query'
		cv2.imshow(query_win, img1_color)
		query_pt_set = []
		cv2.setMouseCallback(query_win, mouseWrapper(query_win, img1_color, collector = query_pt_set))

		train_win = 'train'
		cv2.imshow(train_win, img2_color)
		train_pt_set = []
		cv2.setMouseCallback(train_win, mouseWrapper(train_win, img2_color, collector = train_pt_set))

		cv2.waitKey()
		cv2.destroyAllWindows()

		assert len(query_pt_set) == len(train_pt_set)
		result = json.dumps([query_pt_set, train_pt_set])

		fp = open('a.json', 'w')
		fp.write(result)
		fp.close()

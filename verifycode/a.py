#-*-coding:utf-8-*-
import cv2
import sys, getopt
import numpy as np

def scale(hist, high):
	m = hist.max()
	result = hist * (high / m)
	return result.astype(int)

def rotate(src, angle, border_color):
	height, width = src.shape
	center = (width / 2, height / 2)
	r = cv2.getRotationMatrix2D(center, angle, 1.0)
	rotated = cv2.warpAffine(src, r, (width, height), borderValue = border_color)
	return rotated

if __name__ == '__main__':
	src_url = None
	output_url = None
	opts, args = getopt.getopt(sys.argv[1:], \
		'', \
		['src=', 'output='])
	
	for o, a in opts:
		if o in ('--src'):
			src_url = a
		elif o in ('--output'):
			output_url = a

	assert src_url

	img = cv2.imread(src_url, 0)
	# 使用OTSU自动计算出二值化的阈值
	thresh, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	# 去除左边和上边的横线
	img[:, 0] = 255
	img[0, :] = 255

	# 统计纵向的扫描线上像素分布情况
	histogram = np.zeros((1, img.shape[1]))
	for i in xrange(histogram.shape[1]):
		col = img[:, i]
		histogram[0, i] =  np.count_nonzero(col)
	img_with_hist = img.copy()
	histogram = scale(histogram, 255)

	img_with_hist[-1, :] = histogram[0, :]
	cv2.imwrite(output_url, img_with_hist)

	element = cv2.getStructuringElement(cv2.MORPH_CROSS,(1,3))
	erosion = cv2.dilate(img, element, iterations = 1)
	cv2.imwrite('skel.png', erosion)

	# 找出黑色像素的坐标
	nonblank = np.nonzero(img == 0)
	pt = np.vstack(nonblank).transpose()

	# 进行kmeans聚类
	a, labels, means = cv2.kmeans(pt.astype(np.float32), 4, \
		criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 1, 10), \
		attempts = 1, \
		flags = cv2.KMEANS_PP_CENTERS)

	print a
	print means

	# 将分类用颜色在灰度图上展示
	labels = (255 - (labels + 1) * (255 / 5)).astype(np.uint8)
	blank_image = 255 - np.zeros(img.shape, dtype = np.uint8)
	blank_image[nonblank] = labels.ravel()
	cv2.imwrite('kmeans.png', blank_image)	

	dst = cv2.cornerHarris(img,2,3,0.04)
	img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
	img_color[dst > 0.6 * dst.max()] = (0, 0, 255)
	cv2.imwrite('corner.png', img_color)

	re = rotate(img, -20, 255)
	cv2.imwrite('rotate.png', re)

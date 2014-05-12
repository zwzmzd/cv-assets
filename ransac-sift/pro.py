#-*- coding:utf-8 -*-

import cv2
import numpy as np
import itertools

if __name__ == '__main__':
	img = cv2.imread('aero1.jpg')

	detector = cv2.FeatureDetector_create('SIFT')
	descriptor = cv2.DescriptorExtractor_create('SIFT')

	skp = detector.detect(img)
	skp, sd = descriptor.compute(img, skp)


	flann_params = dict(algorithm=1, trees=4)
	flann = cv2.flann_Index(sd, flann_params)
	idx, dist = flann.knnSearch(td, 1, params={})
	del flann
	



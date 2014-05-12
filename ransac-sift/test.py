#-*-coding:utf-8-*-
import os
import cv2
import numpy as np

OUTPUT_DIR = 'output/.data'

def prepare():
	if not os.path.exists(OUTPUT_DIR):
		os.makedirs(OUTPUT_DIR)

if __name__ == '__main__':
	prepare()

	img = cv2.imread('data/aero1.jpg')
	gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	sift = cv2.SIFT()
	kp = sift.detect(gray,None)

	img=cv2.drawKeypoints(gray,kp,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	cv2.imwrite(os.path.join(OUTPUT_DIR, 'sift_keypoints.jpg'), img)
	cv2.imshow('test', img)
	while True:
		ch = 0xff & cv2.waitKey()
		if ch == 27:
			break

	cv2.destroyAllWindows()

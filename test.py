import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import utils

image = cv2.imread( './test_images/straight_lines1.jpg')
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
hls = cv2.split( cv2.cvtColor(image, cv2.COLOR_BGR2HLS) )
hue = hls[0]
sat = hls[2]

gray = cv2.equalizeHist(gray, gray)
hue = cv2.equalizeHist(hue, hue)
sat = cv2.equalizeHist(sat, sat)


img = gray
step = 10
range_from = 0
range_to = 100
for i in range(range_from,range_to,step):
    max_limit = 255
    max_limit_rad = 2*np.pi

    th_min = max_limit * i / 100
    th_max = max_limit * (i + step) / 100

    th_min_rad = (max_limit_rad * i / 100 ) - np.pi
    th_max_rad = (max_limit_rad * (i + step) / 100 ) - np.pi

    print("Step: {:.0f} {:.0f} \t| Px: {:.0f}  {:.0f} \t| RAD: {:.3f} {:.3f}".format(i,i+step,th_min, th_max, th_min_rad, th_max_rad))

    sx = utils.gradientAbolutes(hue, orient='x', thresh=(th_min, th_max))
    sy = utils.gradientAbolutes(hue, orient='y', thresh=(th_min, th_max))
    mag = utils.gradientMagnitude(hue, thresh=(th_min, th_max))
    dir = utils.gradientDirection(gray, thresh=(th_min_rad, th_max_rad))
    utils.showImages((sx,sy,mag,dir), cmap='gray')



'''
utils.showImages((hue_sx, hue_sy, hue_mag, hue_dir), cmap='gray')


test_image_original = cv2.imread( './test_images/straight_lines1.jpg')
test_image_original = cv2.cvtColor(test_image_original,cv2.COLOR_BGR2RGB)
plt.imshow(test_image_original, cmap="gray")
plt.show()
plt.close('all')

test_image_original = cv2.imread( './test_images/straight_lines2.jpg')
test_image_original = cv2.cvtColor(test_image_original,cv2.COLOR_BGR2RGB)
plt.imshow(test_image_original, cmap="gray")
plt.show()
plt.close('all')

'''

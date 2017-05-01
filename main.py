import cv2
from camera import Camera
import utils
import glob, os
import matplotlib.pyplot as plt
import numpy as np

test_images_path = './test_images/'
calibration_path = './camera_cal/'




image_paths=glob.glob(calibration_path+'*.jpg')
camera = Camera()
camera.calibrate(image_paths)

test_images_paths = glob.glob(test_images_path+'*.jpg')
image = cv2.imread(test_images_paths[0])
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

image = camera.unsidtort(image)
#if debug: utils.showImage(image)
image = camera.threasholdLaneLines(image,debug=True)
utils.showImage(image)
image = camera.ROI(image)
#if debug: utils.showImage(image)
image, debug_image = camera.birdsEye(image, debug=True)
utils.showImage(debug_image)
utils.showImage(image)

camera.processVideo('project_video.mp4',debug=True)






'''
test_images = list(map(lambda image_path:cv2.imread(image_path),test_images_paths))
test_images = list(map(lambda image:cv2.cvtColor(image,cv2.COLOR_BGR2RGB),test_images))
test_images_grid = list(map(lambda image:utils.drawGrid(image),test_images))

test_images_undist =  list(map(lambda img: camera.unsidtort(img),test_images_grid))

interlaved = []
for i in range(len(test_images)):
    interlaved.append(test_images_grid[i])
    interlaved.append(test_images_undist[i])

utils.showImages(interlaved)
'''

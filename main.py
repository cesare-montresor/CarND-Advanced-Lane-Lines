import cv2
from camera import Camera
import utils
import glob, os
import matplotlib.pyplot as plt
import numpy as np

calibration_path = './camera_cal/'

test_images_path = './test_images/*.jpg'
video_frame_path = './frames/*_1000_*.jpg'

debug = False
test_pipeline = False
test_pipeline_path = test_images_path
#test_pipeline_path = video_frame_path


image_paths=glob.glob(calibration_path+'*.jpg')
camera = Camera()
camera.calibrate(image_paths)

if test_pipeline:
    images = utils.loadImages(test_pipeline_path, cv2.COLOR_BGR2RGB)
    hud = camera.pipeline(images[0], debug=debug, dump_partials=False)
    utils.showImage(hud)
else:
    camera.processVideo('project_video.mp4', debug=debug, live=False)






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

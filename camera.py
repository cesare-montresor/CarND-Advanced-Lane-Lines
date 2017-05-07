import cv2
import pickle, os
import numpy as np
import matplotlib.pyplot as plt
import utils
import road
from moviepy.editor import VideoFileClip


default_calibration_file='./camera_params.p'
video_output_path = './videos/'
frame_dump_folder = './frames/'
output_images_pipeline = './output_images/'


class Camera():

    def __init__(self, filename=None):
        if filename is None:
            filename = default_calibration_file
        self.filename = filename
        self.calibrations = None
        self.loadCalibrations()
        self.lastBirdEyePoints = {}
        self.lastLane = []
        self.avgLastN = 3

        self.extractFramePath = None

        ## Calibration

    def calibrate(self, image_paths, size=(9, 6), force=False, debug=False):

        if os.path.isfile(self.filename):
            return True

        nx = size[0]
        ny = size[1]

        # Arrays to store object points and image points from all the images.
        object_points = []  # 3d points in real world space
        image_points = []  # 2d points in image plane.

        # preparing template to use for every object_points
        object_point_template = np.zeros((nx * ny, 3), np.float32)
        object_point_template[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

        # Step through the list and search for chessboard corners
        for idx, filename in enumerate(image_paths):
            img = cv2.imread(filename)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, image_corners = cv2.findChessboardCorners(gray, (nx, ny), None)

            # If found, add object points, image points
            if ret == True:
                # print(image_corners)
                object_points.append(object_point_template)
                image_points.append(image_corners)

                # Draw and display the corners
                if debug:
                    img_lines = cv2.drawChessboardCorners(img, (nx, ny), image_corners, ret)
                    plt.imshow(img_lines)
                    plt.show()
                    plt.close('all')

            else:
                print('Corners NOT found for image', filename)

        sample_image = cv2.imread(image_paths[0])
        height, width, channels = sample_image.shape

        # Do camera calibration given object points and image points
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, (width, height), None, None)

        self.calibrations = {
            'ret': ret,
            'mtx': mtx,
            'dist': dist,
            'rvecs': rvecs,
            'tvecs': tvecs
        }
        # print(self.calibrations)

        if force or not os.path.isfile(self.filename):
            self.saveCalibrations()

        return ret

    def saveCalibrations(self, filename=None):
        if filename is None:
            filename = self.filename
        with open(filename, 'wb+') as picklefile:
            pickle.dump(self.calibrations, picklefile)

    def loadCalibrations(self, filename=None):
        if filename is None:
            filename = self.filename
        if os.path.isfile(filename):
            with open(filename, 'rb') as picklefile:
                self.calibrations = pickle.load(picklefile)

## Video Processing

    def exctractFrames(self, path):
        strdate = '_%04d_' + utils.standardDateFormat()
        output_frame_path = frame_dump_folder + utils.filename_append(path, strdate)
        output_frame_path = utils.change_ext(output_frame_path, 'jpg')
        video = VideoFileClip(path)
        video.write_images_sequence(output_frame_path)


    def processVideo(self, path, live=False, debug=False):
        if not live:
            strdate = '_'+utils.standardDateFormat()
            output_video = video_output_path + utils.filename_append(path,strdate)
            video = VideoFileClip(path)
            video_clip = video.fl_image(self.pipeline)
            video_clip.write_videofile(output_video, audio=False)
        else:
            vidcap = cv2.VideoCapture(path)
            while True:
                success, image = vidcap.read()
                if not success:
                    break
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                final_image = self.pipeline(image,debug=debug)
                utils.showImage(final_image)

## Frame Processing

    def pipeline(self, image, debug=False, dump_partials=False):
        original = image.copy()
        image_unsidtort = self.unsidtort(image)
        if debug: utils.showImage(image_unsidtort)

        image_thresh = self.threasholdLaneLines(image_unsidtort, debug=debug)
        if debug: utils.showImage(image_thresh)

        image_roi = self.ROI(image_thresh)
        if debug: utils.showImage(image_roi)

        image_bird, image_bird_debug = self.birdsEye(image_roi, debug=debug)
        if debug:
            utils.showImage(image_bird_debug)
            utils.showImage(image_bird)

        image_lanes, lane = self.laneSearch(image_bird, debug=debug)
        if debug: utils.showImage(image_lanes)

        final_image = self.HUD(original, debug=debug)
        if debug: utils.showImage(final_image)


        if dump_partials:
            path_tpl = output_images_pipeline + 'pipeline_partials_{:02d}.jpg'
            images = [original, image_unsidtort, image_thresh, image_roi, image_bird, image_lanes, final_image]

            for i, img in enumerate(images):
                path = path_tpl.format(i)
                img = utils.normalizeImage(img)
                cv2.imwrite(path, img)


        return final_image

## Pipeline functions (same order ^_^)


    def unsidtort(self,img):
        undist = cv2.undistort(img, self.calibrations['mtx'], self.calibrations['dist'], None, self.calibrations['mtx'])
        return undist



    def threasholdLaneLines(self,image, kernel_size=(3,2), debug=False):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        hls = cv2.split( cv2.cvtColor(image, cv2.COLOR_RGB2HLS) )

        hue = hls[0]
        lig = hls[1]
        sat = hls[2]

        gray = cv2.equalizeHist(gray, gray)
        #hue = cv2.equalizeHist(hue, hue) # hue should not be equalized (or it would detect a different color)
        sat = cv2.equalizeHist(sat, sat)
        lig = cv2.equalizeHist(lig, lig)

        yellow = np.zeros_like(hue)
        bright = np.zeros_like(gray)
        saturation = np.zeros_like(sat)
        light = np.zeros_like(lig)

        yellow[(hue > 20) & (hue < 30)] = 1
        bright[(gray > 250) ] = 1
        saturation[sat > 240] = 1
        light[lig > 127] = 1

        color = (yellow & light ) | (bright & light ) | (saturation & light)

        if debug:
            utils.showImages((yellow, bright, saturation, light, color), cmap='gray')

        thresh = (100,200)
        hue_sx = utils.gradientAbolutes(hue, orient='x', thresh=thresh)
        gray_sx = utils.gradientAbolutes(gray, orient='x', thresh=thresh)
        sat_sx = utils.gradientAbolutes(sat, orient='x', thresh=thresh)

        hue_sy = utils.gradientAbolutes(hue, orient='y', thresh=thresh)
        gray_sy = utils.gradientAbolutes(gray, orient='y', thresh=thresh)
        sat_sy = utils.gradientAbolutes(sat, orient='y', thresh=thresh)

        if debug:
            utils.showImages((hue_sx , hue_sy , gray_sx , gray_sy , sat_sx , sat_sy), cmap='gray')

        gradient = (hue_sx | gray_sx | sat_sx | hue_sy | gray_sy | sat_sy)

        # density based noise reduction
        # opening: erosion and dilation, see : http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
        kernel = np.ones(kernel_size,np.uint8)
        #kernel = [[0,1,1,1,0],[0,1,1,1,0],[0,1,1,1,0]]
        #kernel = np.array(kernel, np.uint8)
        #gradient_clean = cv2.morphologyEx(gradient, cv2.MORPH_OPEN, kernel, iterations=1)
        # turned out that noise is good for interpolation, i'll try it later down the line


        if debug:
            utils.showImages((color, gradient))
        mask = color #| gradient


        if debug:
            utils.showImages((mask, color, gradient))

        return mask

    def ROI(self, img, offset=None, margin=None ):
        img = img.copy()

        if offset is None:
            offset = (50, 50)

        if margin is None:
            margin = (50, 50)

        h, w = img.shape[0], img.shape[1]
        mask = np.zeros((h,w), np.uint8)

        cx, cy = int(w / 2), int(h / 2)
        left_top, left_bot = (cx - offset[0], cy + offset[1]), (margin[0], h - margin[1])
        right_top, right_bot = (cx + offset[0], cy + offset[1]), (w - margin[0], h - margin[1])
        poly_points = np.array([[left_top, right_top, right_bot, left_bot]], np.int32 )
        #print(type(mask[0]))
        cv2.fillPoly(mask, poly_points, (255,255,255) )
        masked_img = cv2.bitwise_and(img, img, mask=mask)

        return masked_img

    ## valid configurations
    # offset = (75, 100), margin = (200, 70)
    # offset = (75, 105), margin = (200, 40)
    # offset=(65,98), margin=(200,40)
    # offset=(60,95), margin=(200,40)
    # offset=(35,82), margin=(200,40)
    def birdsEye(self,img, offset=(48,88), margin=(200,40), kernel_size=(2,2) ,debug=False):
        h,w = img.shape[0],img.shape[1]
        cx,cy = int(w/2),int(h/2)

        top_left, top_right = [cx - offset[0], cy + offset[1]], [cx + offset[0], cy + offset[1]]
        bottom_left, bottom_right = [margin[0], h-margin[1]], [w-margin[0], h-margin[1]]
        src_point = np.array([top_left, top_right, bottom_right, bottom_left], np.float32)

        top_left[0] = bottom_left[0]   = int( (top_left[0]  + bottom_left[0] )/2)
        top_right[0] = bottom_right[0] = int( (top_right[0] + bottom_right[0])/2)
        top_left[1] = top_right[1] = 0
        bottom_left[1] = bottom_right[1] = h
        dst_point = np.array([top_left, top_right, bottom_right, bottom_left], np.float32)

        self.lastBirdEyePoints = {'src':src_point, 'dst':dst_point}
        M = cv2.getPerspectiveTransform(src_point, dst_point)
        warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)


        # density based noise reduction
        # opening: erosion and dilation, see : http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
        # kernel = np.ones(kernel_size, np.uint8)
        # warped = cv2.morphologyEx(warped, cv2.MORPH_OPEN, kernel, iterations=3)
        # REMOVED: seems that interpolation works better with noise ...
        debug_image = None
        if debug:
            debug_image = warped.copy()*255
            debug_image = cv2.cvtColor(debug_image, cv2.COLOR_GRAY2RGB)
            cv2.polylines(debug_image, np.array([dst_point], np.int), True, (0,255,0))
            utils.showImage(debug_image)


        return warped, debug_image

    def laneSearch(self, binary_image, previous_lane=None, nrow=9, minpix=50, box_margin=50, debug=False):
        h,w = binary_image.shape[0],binary_image.shape[1]

        nonzero = binary_image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        previous_lane = self.lastLane[-1] if len(self.lastLane) > 0 else None
        if previous_lane is not None:
            #print('found previous positions','\n')
            left_position, right_position = previous_lane.left.position, previous_lane.right.position
        else:
            #print('histogram search','\n')
            left_position, right_position = self.laneSearchHistogram(binary_image, debug=debug)
        binary_image, left_info, right_info = self.laneSearchSlidingWindows(binary_image, left_position, right_position, debug=debug)

        left_fit, left_idx = left_info
        right_fit, right_idx = right_info

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_image.shape[0] - 1, binary_image.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        # generated here to be able to include modifications (debug True) done by the lane search functions (hist, window)
        output_img = np.dstack((binary_image, binary_image, binary_image)) * 255
        if debug:
            output_img[nonzeroy[left_idx], nonzerox[left_idx]] = [255, 0, 0]
            output_img[nonzeroy[right_idx], nonzerox[right_idx]] = [0, 0, 255]
            plt.imshow(output_img)
            plt.plot(left_fitx, ploty, color='yellow')
            plt.plot(right_fitx, ploty, color='yellow')
            plt.xlim(0, w)
            plt.ylim(h, 0)
            plt.show()

        left_line = road.Line()
        left_line.fit = left_fit
        left_line.count = len(left_idx)
        left_line.position = left_position

        right_line = road.Line()
        right_line.fit = right_fit
        right_line.count = len(right_idx)
        right_line.position = right_position

        lane = road.Lane()
        lane.left = left_line
        lane.right = right_line

        lane.curvature(debug=debug)
        self.lastLane.append(lane)

        return output_img, lane

    def laneSearchHistogram(self,binary_image,slice_count=2, slice_num=0 ,offset=0,debug=False):
        h, w = binary_image.shape[0], binary_image.shape[1]
        slice_h = int(h/slice_count)

        max_h = h - slice_h * slice_num
        min_h = h - slice_h * (slice_num + 1)

        slice = binary_image[min_h:max_h, :]
        histogram = np.sum(slice, axis=0)
        if debug:
            utils.showImage(slice, cmap='gray')
            print('histogram',histogram.shape)
            plt.plot(histogram)
            plt.show()

        midpoint = int((len(histogram) / 2) + offset)

        left_peak = np.argmax(histogram[:midpoint])
        right_peak = np.argmax(histogram[midpoint:]) + midpoint

        return left_peak, right_peak

    def laneSearchSlidingWindows(self, binary_image, left_position, right_position, nrow = 9, box_margin=None, debug=False, minpix = None):
        if box_margin is None:
            box_margin = 50

        if minpix is None:
            minpix = 50

        h, w = binary_image.shape[0], binary_image.shape[1]

        nonzero = binary_image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        left_lane_idx = []
        right_lane_idx = []


        step = int(np.ceil(h / nrow))
        for row in range(nrow):
            max_y = h - (row * step)
            min_y = max_y - step

            left_box_tl, left_box_br = (left_position - box_margin, min_y), (left_position + box_margin, max_y)
            right_box_tl, right_box_br = (right_position - box_margin, min_y), (right_position + box_margin, max_y)
            if debug:
                binary_image = binary_image.copy()
                cv2.rectangle(binary_image, left_box_br, left_box_tl,  (0, 255, 0), 2)
                cv2.rectangle(binary_image, right_box_br, right_box_tl, (0, 255, 0), 2)

            left_lane_pixels = ( (min_y < nonzeroy) & (nonzeroy < max_y) & (left_box_tl[0] < nonzerox ) & (nonzerox < left_box_br[0]) ).nonzero()[0]
            right_lane_pixels = ((min_y < nonzeroy) & (nonzeroy < max_y) & (right_box_tl[0] < nonzerox ) & (nonzerox < right_box_br[0])).nonzero()[
                0]

            if len(left_lane_pixels) > minpix:
                left_position = np.int(np.mean(nonzerox[left_lane_pixels]))

            if len(right_lane_pixels) > minpix:
                right_position = np.int(np.mean(nonzerox[right_lane_pixels]))

            left_lane_idx.append(left_lane_pixels)
            right_lane_idx.append(right_lane_pixels)

        left_lane_idx = np.concatenate(left_lane_idx)
        right_lane_idx = np.concatenate(right_lane_idx)

        left_x = nonzerox[left_lane_idx]
        left_y = nonzeroy[left_lane_idx]

        right_x = nonzerox[right_lane_idx]
        right_y = nonzeroy[right_lane_idx]

        left_fit = None if len(left_y) == 0 else np.polyfit(left_y, left_x, 2)
        right_fit = None if len(right_y) == 0 else np.polyfit(right_y, right_x, 2)

        return binary_image,(left_fit,left_lane_idx),(right_fit,right_lane_idx)


    def HUD(self, original, debug=False):
        h,w = original.shape[0],original.shape[1]
        if len(original.shape) == 2 or original.shape[2] == 1:
            if np.max(original) <=1: # convert bitmask into image
                original *= 255
            original = cv2.cvtColor(original,cv2.COLOR_GRAY2RGB)

        hud = np.zeros_like(original)

        lastLaneN = self.lastLane[-self.avgLastN:]

        cntN = len(lastLaneN)
        best_fit, left_fit, right_fit = [0,0,0],[0,0,0],[0,0,0]
        best_curve_m, left_curve_m, right_curve_m = 0,0,0
        for lane in lastLaneN:
            best_curve_m += lane.best.curve_rad_m
            left_curve_m += lane.left.curve_rad_m
            right_curve_m += lane.right.curve_rad_m

            best_fit = np.add(best_fit,lane.best.fit)
            left_fit = np.add(left_fit,lane.left.fit)
            right_fit = np.add(right_fit,lane.right.fit)


        best_curve_m = best_curve_m / cntN
        left_curve_m = left_curve_m / cntN
        right_curve_m = right_curve_m / cntN

        best_fit /= cntN
        left_fit /= cntN
        right_fit /= cntN


        ploty = np.array(np.linspace(0, h - 1, num=h), np.int)

        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]


        size = 15
        for i in range(len(ploty)): # need a pythonic implementation here
            y, lx, rx  = int(ploty[i]), int(left_fitx[i]), int(right_fitx[i])
            hud[y, lx - size:lx + size] = (255, 0, 0)
            hud[y, lx + size:rx - size] = (0, 255, 0)
            hud[y, rx - size:rx + size] = (0, 0, 255)

        text = "{:.0f}".format(best_curve_m)
        text_size = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=5, thickness=5)
        text_w,text_h = text_size[0]

        text_orig = ( int(lane.left.position+text_w/2), int(h-20) )
        #print('text_box',text_box,'text_box_width',text_box_width)
        cv2.putText(hud, text, text_orig, fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=5, thickness=5, color=(255, 255, 255))

        src_point,dst_point = self.lastBirdEyePoints['src'], self.lastBirdEyePoints['dst']
        M = cv2.getPerspectiveTransform(dst_point, src_point)
        hud = cv2.warpPerspective(hud, M, (w, h), flags=cv2.INTER_LINEAR)

        merged = cv2.addWeighted(original, 1.0, hud, 0.5, 0)

        return merged

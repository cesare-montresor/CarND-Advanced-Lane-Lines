import cv2
import pickle, os
import numpy as np
import matplotlib.pyplot as plt
import utils
import road

default_calibration_file='./camera_params.p'

class Camera():

    def __init__(self, filename=None):
        if filename is None:
            filename = default_calibration_file
        self.filename = filename
        self.calibrations = None
        self.loadCalibrations()

    def processVideo(self,path,debug=True):
        vidcap = cv2.VideoCapture(path)
        count = 0
        while True:
            success, image = vidcap.read()
            if not success:
                break
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.pipeline(image,debug=debug)


    def pipeline(self, image, debug=False):
        image = self.unsidtort(image)
        #if debug: utils.showImage(image)

        image = self.threasholdLaneLines(image)
        #if debug: utils.showImage(image)

        image = self.ROI(image)
        #if debug: utils.showImage(image)

        image, debug_image = self.birdsEye(image, debug=debug)
        if debug: utils.showImage(debug_image)
        if debug: utils.showImage(image)

        image, lane = self.laneSearch(image, debug=debug)
        #if debug: utils.showImage(image)

        lane.curvature(debug=debug)

    def calibrate(self, image_paths, size=(9,6), test_image=None, force=False, debug=False ):

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
            img = cv2.imread(filename )
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, image_corners = cv2.findChessboardCorners(gray, (nx, ny), None)

            # If found, add object points, image points
            if ret == True:
                #print(image_corners)
                object_points.append(object_point_template)
                image_points.append(image_corners)

                # Draw and display the corners
                if debug:
                    img_lines = cv2.drawChessboardCorners(img, (nx, ny), image_corners, ret)
                    plt.imshow(img_lines)
                    plt.show()
                    plt.close('all')

            else:
                print('Corners NOT found for image',filename)

        sample_image = cv2.imread(image_paths[0])
        height, width, channels = sample_image.shape

        # Do camera calibration given object points and image points
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, (width,height), None, None)

        self.calibrations = {
            'ret':ret,
            'mtx':mtx,
            'dist':dist,
            'rvecs':rvecs,
            'tvecs':tvecs
        }
        #print(self.calibrations)

        if force or not os.path.isfile(self.filename):
            self.saveCalibrations()

        return ret

    def unsidtort(self,img):
        undist = cv2.undistort(img, self.calibrations['mtx'], self.calibrations['dist'], None, self.calibrations['mtx'])
        return undist

    def ROI(self, img, offset=None, margin=None ):
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
        print(type(mask[0]))
        cv2.fillPoly(mask, poly_points, (255,255,255) )
        masked_img = cv2.bitwise_and(img, img, mask=mask)

        return masked_img

    def laneSearchHistogram(self,binary_image,offset=0,debug=True):
        h, w = binary_image.shape[0], binary_image.shape[1]
        slice = binary_image[:int(h/2), :]
        utils.showImage(slice,cmap='gray')
        histogram = np.sum(slice, axis=0)
        if debug:
            print('histogram',histogram.shape)
            plt.plot(histogram)
            plt.show()

        midpoint = int((len(histogram) / 2) + offset)

        left_peak = np.argmax(histogram[:midpoint])
        right_peak = np.argmax(histogram[midpoint:]) + midpoint

        return left_peak, right_peak

    def laneSearchSlidingWindows(self, binary_image, left_position, right_position, nrow = 9, box_margin=None, debug=False, minpix = None):
        if box_margin is None:
            box_margin=50

        if minpix is None:
            minpix = 50

        nonzero = binary_image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        left_lane_idx = []
        right_lane_idx = []

        h, w = binary_image.shape[0], binary_image.shape[1]
        print(binary_image.shape)
        step = int(np.ceil(h / nrow))
        for row in range(nrow):
            max_y = h - (row * step)
            min_y = max_y - step

            left_box_tl, left_box_br = (left_position - box_margin, min_y), (left_position + box_margin, max_y)
            right_box_tl, right_box_br = (right_position - box_margin, min_y), (right_position + box_margin, max_y)
            if debug:
                binary_image = binary_image.copy()
                print()
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

        left_fit = np.polyfit(left_y, left_x, 2)
        right_fit = np.polyfit(right_y, right_x, 2)

        return binary_image,(left_fit,left_lane_idx),(right_fit,right_lane_idx)

    def laneSearch(self, binary_image, previous_lane=None, nrow = 9, minpix= 50, box_margin = 50, debug=False):
        nonzero = binary_image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])


        if previous_lane is not None:
            print('found previous positions')
            left_position, right_position = previous_lane.left.position, previous_lane.right.position
        else:
            print('histogram search')
            left_position, right_position = self.laneSearchHistogram(binary_image,debug=debug)

        binary_image,left_info,right_info = self.laneSearchSlidingWindows(binary_image, left_position, right_position, debug=debug)

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
            plt.xlim(0, 1280)
            plt.ylim(720, 0)
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

        return output_img, lane



    def threasholdLaneLines(self,image, debug=False):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        hls = cv2.split( cv2.cvtColor(image, cv2.COLOR_RGB2HLS) )
        hsv = cv2.split( cv2.cvtColor(image, cv2.COLOR_RGB2HSV) )
        yuv = cv2.split( cv2.cvtColor(image, cv2.COLOR_RGB2YUV) )

        hue = hls[0]
        sat1 = hls[2]
        sat2 = hsv[1]

        #if debug:
        #    utils.showImages((gray, hue, sat1, sat2), cmap='gray')

        gray = cv2.equalizeHist(gray, gray)
        hue = cv2.equalizeHist(hue, hue)
        sat1 = cv2.equalizeHist(sat1, sat1)
        sat2 = cv2.equalizeHist(sat2, sat2)

        #if debug:
        #    utils.showImages((gray, hue, sat1, sat2), cmap='gray')

        yellow = np.zeros_like(hue)
        bright0 = np.zeros_like(gray)
        bright1 = np.zeros_like(sat1)
        bright2 = np.zeros_like(sat2)

        yellow[( hue > 50 )&( hue < 60 )] = 1
        bright0[gray > 240] = 1
        bright1[sat1 > 240] = 1
        bright2[sat2 > 240] = 1
        color = ((bright0 & bright1) | (bright0 & bright2)) | yellow

        if debug:
            utils.showImages((yellow, bright0, bright1, bright2, color), cmap='gray')

        hue_sx = utils.gradientAbolutes(hue, orient='x', thresh=(40, 255))
        hue_sy = utils.gradientAbolutes(hue, orient='y', thresh=(60, 255))
        hue_mag = utils.gradientMagnitude(hue, thresh=(50, 255))
        hue_dir = utils.gradientDirection(gray, thresh=(0.8, 0.9))

        if debug:
            utils.showImages((hue_sx, hue_sy, hue_mag, hue_dir), cmap='gray')

        gray_sx = utils.gradientAbolutes(gray, orient='x', thresh=(40, 255))
        gray_sy = utils.gradientAbolutes(gray, orient='y', thresh=(60, 255))
        gray_mag = utils.gradientMagnitude(gray, thresh=(50, 255))
        gray_dir = utils.gradientDirection(gray, thresh=(0.8, 0.9))

        sat1_sx = utils.gradientAbolutes(sat1, orient='x', thresh=(20, 255))
        sat1_sy = utils.gradientAbolutes(sat1, orient='y', thresh=(60, 255))
        sat1_mag = utils.gradientMagnitude(sat1, thresh=(50, 255))
        sat1_dir = utils.gradientDirection(sat1, thresh=(0.8, 0.9))

        sat2_sx = utils.gradientAbolutes(sat2, orient='x', thresh=(20, 255))
        sat2_sy = utils.gradientAbolutes(sat2, orient='y', thresh=(60, 255))
        sat2_mag = utils.gradientMagnitude(sat2, thresh=(50, 255))
        sat2_dir = utils.gradientDirection(sat2, thresh=(0.8, 0.9))

        avg_s = (hue_sx & hue_sy) | (gray_sx & gray_sy) | (sat1_sx & sat1_sy) | (sat2_sx & sat2_sy)
        avg_mag = (hue_mag & gray_mag  & sat1_mag & sat2_mag)
        avg_dir = (hue_dir & gray_dir & sat1_dir & sat2_dir )

        avg_mask = ( color | yellow | avg_s | avg_mag | avg_dir  )
        if debug:
            utils.showImages((avg_mask, color, yellow, avg_s, avg_mag, avg_dir))
        lines = np.zeros_like(avg_mask)
        lines[avg_mask == 1] = 255



        return lines

    def birdsEye(self,img, offset=(65,100), margin=(200,60), lines_distance=700,debug=False):
        h,w = img.shape[0],img.shape[1]
        cx,cy = int(w/2),int(h/2)

        top_left, top_right = [cx - offset[0], cy + offset[1]], [cx + offset[0], cy + offset[1]]
        bottom_left, bottom_right = [margin[0], h-margin[1]], [w-margin[0], h-margin[1]]
        src_point = np.array([top_left, top_right, bottom_right, bottom_left], np.float32)


        top_left[0] = bottom_left[0]   = int((w/2)-(lines_distance/2))
        top_right[0] = bottom_right[0] = int((w/2)+(lines_distance/2))
        top_left[1] = top_right[1] = 0
        bottom_left[1] = bottom_right[1] = h
        dst_point = np.array([top_left, top_right, bottom_right, bottom_left], np.float32)

        M = cv2.getPerspectiveTransform(src_point, dst_point)
        warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)

        if debug:
            img = img.copy()
            cv2.polylines(img, np.array([src_point], np.int), True, (255,0,0))

        return warped, img

    def saveCalibrations(self,filename=None):
        if filename is None:
            filename = self.filename
        with open(filename,'wb+') as picklefile:
            pickle.dump(self.calibrations,picklefile)


    def loadCalibrations(self,filename=None):
        if filename is None:
            filename = self.filename
        if os.path.isfile(filename):
            with open(filename, 'rb') as picklefile:
                self.calibrations = pickle.load(picklefile)




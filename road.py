import numpy as np
import cv2
import matplotlib.pyplot as plt

class Line():
    def __init__(self):
        self.fit = None
        self.count = 0
        self.position = None

        self.curve_rad_px = None
        self.curve_rad_km = None

        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units

        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None

    def __str__(self):
        display = {
            'fit':self.fit,
            'count':self.count,
        }
        return str(display)

class Lane():
    def __init__(self):
        self.count = None
        self.timestamp = None
        self.left = None
        self.right = None

    def __str__(self):
        display = {
            'count': self.count,
            'timestamp': self.timestamp,
            'left': str(self.left),
            'right': str(self.right),
        }
        return str(display)

    def curvature(self, img_size=(1280,720), margin=50,  ym_per_pix = (30/720), xm_per_pix = (3.7/700), debug=False):
        # pick the line with more points
        w,h = img_size

        # Generate some fake data to represent lane-line pixels
        ploty = np.linspace(0, h-1, num=h)

        left_fit, right_fit = self.left.fit, self.right.fit
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        # For each y position generate random x position within +/-50 pix
        # of the line base position in each case (x=200 for left, and x=900 for right)
        leftx  = np.array([x + np.random.randint(-margin, high=margin+1) for x in left_fitx])
        rightx = np.array([x + np.random.randint(-margin, high=margin+1) for x in right_fitx])

        print('pixels line distance bottom (~700px)', leftx[-1] - rightx[-1])
        print('pixels line distance top (~700px)', leftx[0] - rightx[0])

        if debug:
            # Plot up the fake data
            mark_size = 3
            line_width = 3
            plt.plot(leftx, ploty, 'o', color='red', markersize=mark_size)
            plt.plot(rightx, ploty, 'o', color='blue', markersize=mark_size)
            plt.xlim(0, w)
            plt.ylim(0, h)
            plt.plot(left_fitx, ploty, color='green', linewidth=line_width)
            plt.plot(right_fitx, ploty, color='green', linewidth=line_width)
            plt.gca().invert_yaxis()  # to visualize as we do the images
            plt.show()
            plt.close()

        point_y = h

        self.left.curve_rad_px = ((1 + (2 * left_fit[0] * point_y + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
        self.right.curve_rad_px = ((1 + (2 * right_fit[0] * point_y + right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])

        # Define conversions in x and y from pixels space to meters


        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)
        # Calculate the new radii of curvature

        self.left.curve_rad_m = ((1 + (2 * left_fit_cr[0] * point_y * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute( 2 * left_fit_cr[0])
        self.right.curve_rad_m = ( (1 + (2 * right_fit_cr[0] * point_y * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute( 2 * right_fit_cr[0])

        print(self.left.curve_rad_m, 'm', self.right.curve_rad_m, 'm')


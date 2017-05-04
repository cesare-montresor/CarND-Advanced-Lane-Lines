import matplotlib.pyplot as plt
import math
import numpy as np
import cv2
import datetime

## IMAGE DISPLAY

def showImages(images, cols=None, rows=None, cmap=None):
    if rows is None and cols is None:
        rows = cols = int(math.ceil(math.sqrt(len(images))))
    if rows is None:
        rows = int(math.ceil(len(images) / cols))
    if cols is None:
        cols = int(math.ceil(len(images) / rows))

    if type(images[0]) == type(""):
        images = list(map(lambda image_path:cv2.imread(image_path),images))

    i = 0
    f, sub_plts = plt.subplots(rows, cols)
    for r in range(rows):
        for c in range(cols):
            sub_plts[r, c].axis('off')
            if i<len(images):
                sub_plts[r,c].imshow(images[i],cmap=cmap)
                i += 1

    plt.show()
    plt.close('all')

def showImage(image, cmap=None):
    if type(image) == type(""):
        image = cv2.imread(image)
    plt.imshow(image, cmap=cmap)
    plt.show()
    plt.close()

def drawGrid(img,rows=10,cols=10):
    img = img.copy()
    h,w,d = img.shape
    dh = h / rows
    dw = w / cols
    for r in range(rows):
        for c in range(cols):
            cv2.line(img, (0, int(dh*r)), (w,int(dh*r)), (255, 0, 0), 5) # horizontal
            cv2.line(img, ( int(dw*c), 0), ( int(dw*c), h), (0, 255, 0), 5) # vertical
    return img

## IMAGE MODIFICATION

def cropImage(image,margins): # css style: top, right, bottom, left
    h,w,d = image.shape
    return image[margins[1]:w-margins[3], margins[0]:h-margins[2]]





## IMAGE THRESHOLDING

def gradientDirection(gray_img, sobel_kernel=3, thresh=(np.pi / 8, np.pi / 2), debug=False):
    # 2) Take the gradient in x and y separately
    sx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sy = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sx, abs_sy = np.absolute(sx), np.absolute(sy)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    grad_dir = np.arctan2(abs_sy, abs_sx)
    if debug:
        plt.hist(grad_dir.T.flat,bins=1000)
        plt.show()
        plt.close()
    # 5) Create a binary mask where direction thresholds are met
    grad_dir_bin = np.zeros_like(grad_dir, np.uint8)
    grad_dir_bin[(grad_dir > thresh[0]) & (grad_dir < thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    # binary_output = np.copy(img) # Remove this line
    return grad_dir_bin


def gradientMagnitude(gray_img, sobel_kernel=3, thresh=(100, 200), debug=False):
    # 2) Take the gradient in x and y separately
    sx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sy = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # 3) Calculate the magnitude
    mag = np.sqrt(sx ** 2 + sy ** 2)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_mag = np.uint8(255 * (mag / np.max(mag)))
    if debug:
        plt.hist(scaled_mag.T.flat ,bins=1000)
        plt.show()
        plt.close()
    # 5) Create a binary mask where mag thresholds are met
    binary_mag = np.zeros_like(scaled_mag, np.uint8)

    binary_mag[(scaled_mag >= thresh[0]) & (scaled_mag <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary_mag

def gradientAbolutes(gray_img, orient='x', thresh=(100,200), debug=False):
    x,y = (1,0) if orient=='x' else (0,1)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    sobel = cv2.Sobel(gray_img, cv2.CV_64F, x, y)
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*(abs_sobel/np.max(abs_sobel)))
    if debug:
        plt.hist(scaled_sobel.T.flat ,bins=1000)
        plt.show()
        plt.close()
    binary_sobel = np.zeros_like(scaled_sobel, np.uint8)
    binary_sobel[(scaled_sobel>=thresh[0]) & (scaled_sobel<=thresh[1])] = 1
    # 5) Create a mask of 1's where the scaled gradient magnitude
            # is > thresh_min and < thresh_max
    # 6) Return this mask as your binary_output image
    #binary_output = np.copy(img) # Remove this line
    return binary_sobel

## files handling

def standardDateFormat():
    return datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

def filename_append(path, suffix):
    parts = path.split(".")
    ext = parts[-1]
    base = ".".join(parts[:-1])+suffix+'.'+ext
    return base


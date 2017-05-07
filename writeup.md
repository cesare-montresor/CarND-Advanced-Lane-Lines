## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[original]: ./writeup_images/original.jpg "Original"
[birdeye]: ./writeup_images/birdeye.png "birdeye"
[birdeye_debug]: ./writeup_images/birdeye.png "birdeye_debug"
[calibration_effects]: ./writeup_images/calibration_effects.png "calibration_effects"
[color_gradients]: ./writeup_images/color_gradients.png "color_gradients"
[gradients]: ./writeup_images/gradients.png "gradients"
[curvature_plot]: ./writeup_images/curvature_plot.png "curvature_plot"
[direction_gradient]: ./writeup_images/direction_gradient.png "direction_gradient"
[direction_gratient_test]: ./writeup_images/direction_gratient_test.png "direction_gratient_test"
[final_image]: ./writeup_images/final_image.png "final_image"
[hist_search]: ./writeup_images/hist_search.png "hist_search"
[hist_search_pikes]: ./writeup_images/hist_search_pikes.png "hist_search_pikes"
[roi]: ./writeup_images/roi.png "roi"
[threshold_colors]: ./writeup_images/threshold_colors.png "threshold_colors"
[thresholded]: ./writeup_images/thresholded.png "thresholded"
[undistort]: ./writeup_images/undistort.png "undistort"
[window_search]: ./writeup_images/window_search.png "window_search"
[window_search_interpolation]: ./writeup_images/window_search_interpolation.png "window_search_interpolation"
[project_video_final]: ./project_video_final.mp4 "project_video_final"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Project structure

- **main.py**  
  The main script, just hit run
- **camera.py**  
  Contains all the methods that transform the image as well as the mothos to process the whole video wraped-up in a Camera class. The methos are organized in the (most likely) order of execution. 
- **road.py**  
  Contains the classes of objects detected on the road, currently: Line and Lane. It also compute the radius of the curvature.  
- **utils.py**  
  Contains various helper methods to load, display and transform images. It also contains the methods to compute various gradients. 
- **output_images/**
  Contains the partials images of the pipeline.
- **frames/**
  Used to dump images from a video, useful for testing and tuning the pipeline on unfriendly frames (see: Camera.extractFrames() ) 
- **camera_params.p**
  Store the parameters obtained during the camera calibration is used to undistort images.
  
### Writeup

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

I've have been using the images provided in the `camera_cal/` folder, together with `cv2.findChessboardCorners()`  
I've been collecting the coordinates of the chessboard corners for each image and store the point, I then fed the points collected to the `cv2.calibrateCamera()` function, obtained the distorsion matrix and store it into `camra_params.p` for later use and to avoid to recalculate them every time. (camera.py:32,92,98,172)    
I then tested the obtained matrix on the test images, but as the result feel very "natural" might not be very clear how the images have been modified, I have decided to add a grid to better see the result.  

![calibration_effects]


#### Pipeline & video processing

The code for the pipeline can be found a in the function `pipeline()` (camera.py:134). The pipeline process a single image.  
The code for the video precessing can be found a in the function `processVideo()` (camera.py:115). It extract a single frame an pass it to the `pipeline()` function.
The `processVideo()` can be called with parameter `live=True` to see the result as it goes instead of writing on the disk.
The `pipeline()` can be called using `dump_partials=True` to dump the intermediate images.
Both the functions can be called with a parameter `debug=True` to see intermediate stage of processing.


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Once the camera distorsion is correctly computed, is enough to call the `cv2.undistort()` (camera.py:172)
to obtain the following result

**Original**
![original]

**Undistorted**
![undistort]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The code related to line thresholding can be found in the function `threasholdLaneLines()` (camera.py:176)
I must start by saying that this part is my least favorite of all, as takes forever and deliver a never-good-enough result, so I started off using a mix a color thresholding on various channels and gadients on the all the same channels, hoping to brute force the problem by summing (+) up all the resulting images and then keep only the pixels above a certain threshold, the idea was to use every thresholded channel and gradient to cast a vote on the pixel.
The result was okish and the computation quite long, after many attempts I ended up ditching direction gradients as it is really too noisy and later all the gradients as the were basically not adding more information compared to the colors but they were introducing a lot of noise. 
The current configuration I've been using only color thresholding, on the gray image and the hls colo space.
I've been using the hue channel to "select" yellow pixels and the gray and saturation channels to spot white color.
I later introduced the additional threshold on lightness, to ensure that the saturation channel would pick up only bright colors.

**Threshold colors**  
![threshold_colors]

**Threshold gradients**  
![gradients]

**Color + Gradient**  
![color_gradients]

**Final threshold**  
![thresholded]

Even if most of the scene is cut out by the prespective transformation, I anyway decided to mask the image with a ROI to futher reduce noise near horizon.  

**Final threshold + ROI**  
![ROI]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code related to the perspective transform can be found in the function `birdsEye()` (camera.py:265)  
I started of as suggested by picking points around the lanes in test images, but the result was quite poor, so I've tried to collect several boxes and average them, still a poor result, a few pixel difference near the horizont make huge difference in the final result, my images where all skewed in one way or another.   
So I decided to go for a numeric way and tune the parameter until I reached an acceptable result, for doing so I imagined a trapezoid going from the center of the image to the bottom of the image.     
There are 2 (4 really) parameters to tune the shape and they can be passed to the function:  

`def birdsEye(self,img, offset=(48,88), margin=(200,40), debug=False):`  

- offset:  horizontal and vertical offset from the center of the image  
- margin:  left and bottom margin  

![birdeye]  

At some point I've also considered a way to make it self tuning, by drawing lines of a specific color (removing that color from the image before eventually) performing a transformation, linearly interpolating the lines in the transformed image, and slowly tune the parameters until it reach straight lines, but by then I already reached an acceptable result, next project ^_^  

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I then perform the the line searching using the function  `laneSearch()` (camera.py:239)  
The function initally use `laneSearchHistogram()` (camera.py:348) to find the starting point of the left and right lane.  

![hist_search]
![hist_search_pikes]

Those starting points are then used by the `laneSearchSlidingWindows()` (camera.py:370) to find the line pixels.

![window_search]

and the a polinomio is fit to the pixels

![window_search_interpolation]

As a final step of the `laneSearch()` function (camera.py:329) the information is organized into 2 `Line()` objects (road.py:5), the 2 `Line()` objects are then assigned to a `Lane()` object which represent an abstract version of the lane lines the current frame.  
All the lane object as stored for future use.  
The lane object also compute the radius of curvature (see next point for details)    

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The code related to the computation of the radius of curvature can be found in the function `curvature()` (road.py:79)  
In order to reduce the errors I decided to compute the radius on a 3rd polynomio produced by `Lane()` method `bestLine()` (road.py:61)  
I've been thinking of several way to produce the `bestLine()` and the current solution can be improved in may ways (see last point) however, the current solution perform a weighted average on the first 2 terms of the polinomio based on the amount of pixels used to fit the original polinomios.     
As for the 3 term I used a simple average to place it in the middle (even if it doesn't really matter it's x position for computing the curvature... but it looks better)  
To reduce stabilize the computation of the radius I also reduced to 1 the margin of the random point used to interpolate the circle, that's why in the plot the dots are not really visible.  
In order to bring transform from px space to m space I've been using the distance in pixels between the lane for the x value and for the y values I've been using the size of the segmented lines from the birdeye view(manually) to estimate the overall length of the observed area.   
 
![curvature_plot]

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The code that produce the overlay on the original image in the function `HUD()` (camera.py:427)  
I first generate the overlay on an empty image for the lane (green) giving evidence to the left(red) and right(blue) lanes, also the radius curvature is printed in the middle (almost, cv2.getTextSize() not sure ) of the lane.    
Than using previously stored transformations points generated by `birdEye()` function (camera.py:279) I reverse the prospective transformation and merge it with the original image.  
In order to smooth the result the `HUD()` function average the last N (3, Camera.avgLastN) polynomios as well as the radius (better ways can be used, see last point)  

![final_image]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_final.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I feel slightly uncomfortable with this project as the code contains an impressive amount of hardcoded parameters, which is already a clear signal of how the solution have been tailored on this one problem and it will miserably fail on slightly different scenario.  
Above all, as for the first project in this course, the weakest spot is the method by which the lane lines pixels are extracted from the image, the thresholding on colors and gradients.  
I've been playing around a lot with it (see test_gradient.py, test_gradient_dir.py), by extracting all the frames of the video (camera.py:107) and fine tune it on faulty frames however it still remain a quite delicate system.  

If I could solve this problem the way I wanted I would tried the following approach: 
- record a video of a driving car in the best condition possible.
- use classing computer vision techniques to build a dataset
- heavily augment the dataset with a combination of the following:
  - hue variation 
  - discoloration
  - contrast adjustments
  - brightness adjustments 
  - dark shadows 
  - light shadow (excess of light)
  - solid occlusions (for other vehicles)
  - horizontal and vertical transitions (slopes)
  - horizontal and vertical image multiple skewing (series of curves)
- train a (small?) convNet for line thresholding and lane searching (together ? separately ? separately and trained together?)   
- used the model to produce more dataset from video of driving in worst condition, cleanup the dataset using classing computer vision tecniques, repeat the whole process.

I believe is much more fast and easy and deliver a more robust result to augment a dataset for every possible variation then writing a software that is flexible enough to accomodated every variation of the road.   
More over, I believe that using this kind of approach, the dataset can be easily extended and the model retrained to include other types of terrain (drit roads, ) or driving conditions (night, snow, rain, fog, etc) as long as a left and right margin can be identified.  

Also a great improvements and robustness can be obtained by "stitching" the lanes pixels from the previous frames with the current one and fit the polynomio on the resulting image, however is not trivial as the due to the prospective transformation the 2 images will never match perfectly like in a 360° image. Perhaps using the speed of the car, or an estimation of it (GPS, segmented lines?) 2 frames can be easily overlayed.  

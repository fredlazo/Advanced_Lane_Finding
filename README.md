## The output video (road with lane line area marked) is project_output.mp4.

## findlines.py contains the video processing pipeline.

## functions.py contains utility functions.

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

[dist_and_undist]: ./output_images/dist_and_undist.png "Distorted and undistorted images"
[filters]: ./output_images/filters.png "Filters"
[pipeline]: ./output_images/pipeline.png "Pipeline"
[testimage]: ./output_images/testimage.png "More challenging test image"



### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.


Please refer to the following python files:

* **function.py**, `distortion_coefs()` (line 8)
* **findlines.py**, `distortion_coefs()` (line 12)

**Camera Calibration**
I used `cv2.findChessboardCorners()` to locate the corners of squares -- defined as the point where two black and two white squares meet -- on chessboard images taken with the camera, then appended the coordinates of these corners to the array `imgpoints`. To the array `objpoints`, I added regularly spaced coordinates where the corners should lie if the image is centered with a particular scaling. Distortion coefficients are determined by comparing `objpoints` and `imgpoints` using `cv2.drawChessboardCorners()`.  This step of determining the distortion coefficients is only performed once before proceeding with the pipeline.

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Using the distortion coefficients
provided above, the pipeline applied `cv2.undistort()` to undistort an image.

![Distorted and undistorted images][dist_and_undist]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

**functions.py**, **`apply_filters()`****(line 37):**  this is where I create a thresholded binary image with lane line pixels tagged as 1's.

The binary output is produced by combining several filters as follows, schematically:

`lightness filter & (( gradient mag filter & gradient angle filter) | saturation filter)`

In the pipeline, `apply_filters()` is invoked at **line 23** of **findlines.py**.

The following sequence shows simple test cases of pixels tagged by each of the filterdistortion_coefss and the resulting binary output.  

![Filters and resulting binary output][filters]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

I used **`cv2.getPerspectiveTransform()`** to find the transformation matrix at **line 45** of 
**findlines.py**, then applied the transformation using **`cv2.warpPerspective()`** at **line 49**.

I used these hardcoded source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| [1121, 720]   | [1121, 720] 
| [ 209, 720]   | [ 209, 720]
| [ 592, 450]   | [ 209,   0]
| [ 692, 450]   | [1121,   0]

This resulted in straight lane lines becoming approximately vertical in the transformed image.

For examples of pre and post warp frames, see section 6, **"Provide an example image of your result plotted back down onto the road such that the lane area is identified clearlypoint 6 below"** 

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?


Lane-line pixels are identified in **`fit_lane_lines()`**, at **line 122** of **functions.py**.  `fit_lane_lines()` calls `find_window_centroids()`
at **line 88** of **functions.py** to perform the convolution search described below.  `fit_lane_lines()` is called at **line 52** of **findlines.py** within the processing pipeline.


Using a convolution technique, I identified left and right lane line pixels in the warped, filtered binary image .
A horizontal portion -- the bottom 1/4 of the warped image -- was summed vertically and the resulting 1D-array convolved with an array 
of 1s of width=150.  The convolution showed peaks in the area of the left and 
right lane lines.  This procedure was repeated for the remaining three horizontal portions.

I used a convolution width of 150 because the perspective transform tended to
warp regions of suspected lane line pixels to 100+ pixels across near the top of the image. Since the gradient and color filters applied earlier successfully filtered for lane line pixels alone,  using a large width of 150 did not create false detection.

After the left and right lane line pixels were identified using convolutions, the locations of the pixels found in those windows
were used to fit two quadratic polynomials, one for each lane.

An example of the lane line regions identified by convolutions for a warped+filtered image, along with the corresponding
quadratic fits, can be seen in the image sequence below.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I calculated he radius of curvature by generating a set of xy coordinates along each lane line polynomial in pixel space,
converting them to world space (convert pixel to meters), fitting a new polynomial to the
pixels in world space, and applying a formula to the world-space polynomial coefficients at the car's location 
(bottom of of the image).  This is implemented in `get_radii_of_curvature()` at **line 233** of **functions.py**.  Within the pipeline, 
`get_radii_of_curvature()` is invoked at **line 58** of **findlines.py**. 

The position of the vehicle with respect to center is computed by calculating the center of the detected lane region at the bottom of
the image, as the average of the left and right polynomials evaluated at the bottom of the image.  This value is subtracted from
the width of the image/2 to obtain an off-center distance in pixels, which is then multiplied by the x-direction pixel-to-meters 
conversion factor of 3.7/700.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

In `draw_unwarped_lane_region()`,  **line 202**
of **functions.py**, the lane line region is drawn, and warped back onto the undistorted image. `draw_unwarped_lane_region()` is invoked at **line 55** of **findlines.py** in the pipeline.

Below are images of the from my pipeline, from filtered image to drawing the final frame.

![Pipeline][pipeline]

This is my pipeline operation on a more challenging test image.

![More challenging test image][testimage]

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here is a [link to my video result](./project_output.mp4).

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The most challenging part of this project was using a filter that consistently detected lane line pixels in various lighting & background
conditions for both yellow and white lane lines.
A combination of saturation, lightness, gradient magnitude, and gradient direction gave reliable detection of lane lines 
on the road regions of the** image.  You  can see my final filter choices at `apply_filters()`  **line 37** of **functions.py**.

At greater curvatures and longer distances, my pipeline tends to fail, as can be seen in the image above.  The highlighted region doesn't detect the right lane.  When the road curves too far outside the trapezoidal source region used for the perspective transform, the pixels that
veer too far from the trapezoidal region are warped out of the transformed image, and end up not being used in the convolution
search or polynomial fit.  Using a less aggressive perspective transform (a wider trapezoidal
source region that gives the road more room to curve) may lessen the effect.

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

[image1]: ./examples/undistort_output.jpg "Undistorted"
[image2]: ./examples/test2.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[image7]: ./examples/hls_channels.jpg "HLS Channels"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the `calibrate.py`

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. Both camera matrix and distortion coefficients are stored in the pickle file "Distortion.p" and will be read in the "pipeline.py", in which the pipeline is implemented

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color in HLS color space and gradient thresholds to generate a binary image (thresholding steps at lines 142 through 155 in `pipeline.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes in the 3rd stage in the pipepline() function, which appears in lines 404 through 406 in the file `pipeline.py`.  It takes as inputs an undistorted image (`undist`), as well as source (`src_pts`) and destination (`dst_pts`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32([(237, 690), (1043, 690), (682, 450), (590, 450)])
dst = np.float32([(325, 720), (955, 720), (955, 0), (300, 0)])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 237, 690     | 325, 720       |
| 1043, 690      | 955, 720      |
| 682, 450     | 955, 0     |
| 590, 450      | 300, 0        |

I verified that my perspective transform was working as expected by drawing the `src_pts` and `dst_pts` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 135 through 139 in my code in `pipeline.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 371 through 391 in my code in `pipeline.py` in the function `lane_plot()` to plot lanes in the warped images. Then in lines 420 through 439 I applied reverse transform to acquire the images with lanes in the normal perspective of view and put texts onto the images

Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_video/out_project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Apart from what has been covered above, a couple of issues appeared during this project and I came up with fixes towards them as following:

1. First problem I met during this project is the warped image has better quality at the bottom than the top. It was due to applying perspective transform will magnify any noise the detection errors at further spots. Therefore during the sliding windows instead of applying a constant width sweeping windows, I increase the window width from 50 by default to 100 linearly as moving from bottom to top (line 211 and 212 in `pipeline.py`)

2. In `slide_window()` I implemented an instant inspection after sweeping each layer (lines 235 to 265 in `pipeline.py`). It looked into the intensity of the maximal window together with the distance between left and right lane segments. If any of these 2 results are not rational, averaged coefficients from past `num_it` of valid detections will be adopted to predict the position of left and right lane centers. Large difference between predicted and actual result indicates invalid center detection and this layer will not be appended in the `allx` list for the corresponding lane. After scanning over all layers, only the lane with sufficient valid centers detected (70% of total layers) will be considered as a valid lane.

3. Second issue I found was the noise might influence the detection when line was not clearly detected, especially when the lane has clear boundary outside of road shoulder, or when detecting dashed lane instead of solid line. Therefore I wrote function `lane_valid()` (line 279 to 444 in `pipeline.py` to apply sanity check and necessary fix towardes the results from detection. It consists of 3 steps:

    i) Primary inspection: residues from polynomial fit was adopted as the criteria in this step. Lane detection results with residues higher than threshold will be marked as invalid (lines 283 to 303 in `pipeline.py`)

    ii) Cross compare: Results from left and right lanes will be compared with respect to difference from the previous `num_it` of valid dectection and radius of curvature. If difference between left and right lanes are greater than specified threshold, both lane detections will be marked as invalid; If one of the lane failed the primary inspection or any other checks, results from the other valid lane will be used to recover the failed one (lines 309 to 338 in `pipeline.py`)

    iii) Final action: valid detection will be recorded and used to calculate the average results from last `num_it` of valid detection. On the other hand invalid detection will be discarded and the averaged one will be adopted (lines 343 to 444 in `pipeline.py`)

4. In the project video, I found program is likely to fail when shadow covers majority of the road. It turns out that saturation channel in HLS color space is sensitive to shadow as well. Also I found the L channel with a proper threshold can filter out the areas with shadow effectively. Thus I introduce the L channel in the filter as well. Results from both H and L channels are further combined with "and" logic operation to eliminate the effects of shadows. Below is the images with L+S channel filters and with only S channel filters

![alt text][image7]

Based on the result file from project video, the program is most likely to fail when vehicle was dring across non-flattened surface since the position of camera will moved when vehicle moves in vertical direction and it will cause further distortion on the images. Even though the program will not use results from these frames, the lanes plotted might not fit the real one well due to any slight curvature changes during these frames.

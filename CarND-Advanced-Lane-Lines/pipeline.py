# Udacity Autonomous Driving Nano Degree Term 1 Project 4: Advanced Lane Finding
# Created by Fengwen Song
# This is the implementation of image processing pipe line

import cv2
import sys
import pickle
import numpy as np
from moviepy.editor import VideoFileClip


class Line:
    def __init__(self):
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
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None


cam_mtx = 0
dist_coef = 0


def abs_sobel_thresh(img, orient='x', thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    abs_sobel = 0
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    # Return the result
    return binary_output


def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Calculate the magnitude
    sobelxy = np.sqrt((sobelx * sobelx) + (sobely * sobely))
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    sobelxy_scaled = np.uint8(255 * sobelxy / np.max(sobelxy))
    # 5) Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(sobelxy_scaled)
    # 6) Return this mask as your binary_output image
    binary_output[(sobelxy_scaled >= thresh[0]) & (sobelxy_scaled <= thresh[1])] = 1
    return binary_output


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    # Apply the following steps to img
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.abs(sobelx)
    abs_sobely = np.abs(sobely)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    grad = np.arctan2(abs_sobely, abs_sobelx)
    # 5) Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(grad)
    # 6) Return this mask as your binary_output image
    binary_output[(grad >= thresh[0]) & (grad <= thresh[1])] = 1
    return binary_output


def hls_select(img, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output


def lane_filter(image):
    ksize = 5
    # Create a grayscale copy of the undistorted image
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gradx = abs_sobel_thresh(gray, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    grady = abs_sobel_thresh(gray, orient='y', sobel_kernel=ksize, thresh=(20, 100))
    mag_binary = mag_thresh(gray, sobel_kernel=ksize, mag_thresh=(30, 100))
    dir_binary = dir_threshold(gray, sobel_kernel=ksize, thresh=(0.7, 1.3))
    his_binary = hls_select(image, thresh=(170, 255))
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (his_binary == 1)] = 1
    return combined


def slide_windows(warped, lanes):
    left = lanes[0]
    right = lanes[1]
    return left, right


def pipeline(image):
    # This is the pipeline function used to process each image passed by main function and get information including
    # Position of lanes, curvature and vehicle position relative to lanes
    # Define 2 line objects for the left and right lanes
    lane_left = Line()
    lane_right = Line()
    src_pts = np.float32([(280, 675), (1042, 675), (614, 435), (666, 435)])
    dst_pts = np.float32([(300, 720), (980, 780), (300, 0), (980, 0)])

    # 1st stage, get undistorted images
    undist = cv2.undistort(image, cam_mtx, dist_coef, None, cam_mtx)

    # 2nd stage, filter out the possible lanes in the images
    filtered = lane_filter(undist)

    # 3rd stage, apply perspective transform to the image
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    M_inv = cv2.getPerspectiveTransform(dst_pts, src_pts)
    warped = cv2.warpPerspective(filtered, M, (filtered.shape[1], filtered.shape[0]), flags=cv2.INTER_LINEAR)

    # 4th stage, Adopt convolution method to determine details about lane
    lane_left_new, lane_right_new = slide_windows(warped, [lane_left, lane_right])

    # 5th stages, apply reverse perspective transform to get images with detected lanes and statistics plotted
    output = cv2.warpPerspective(filtered, M_inv, (filtered.shape[1], filtered.shape[0]), flags=cv2.INTER_LINEAR)
    return output


if __name__ == "__main__":
    if len(sys.argv) == 3:
        f_input = sys.argv[1]
        f_output = sys.argv[2]
    elif len(sys.argv) == 2:
        f_input = sys.argv[1]
        f_output = ''
    else:
        print('Incorrect calling format')
        sys.exit(-2)
    # Load cam matrix and distortion coefficients
    dist_res = pickle.load(open('Distortion.p', 'rb'))
    cam_mtx = dist_res['Cam Mat']
    dist_coef = dist_res['Dist Coeff']
    # Determine the type of input files (images/videos)
    file_input = f_input.split('/')
    output_file = f_output + '/out_' + file_input[-1]
    if '.jpg' in file_input[-1]:
        img_in = cv2.imread(file_input)
        img_out = pipeline(img_in)
        cv2.imwrite(output_file, img_out)
    elif '.mp4' in file_input[-1]:
        clip1 = VideoFileClip(file_input)
        lane_clip = clip1.fl_image(pipeline)
        lane_clip.write_videofile(output_file, audio=False)
    else:
        print('Not a valid input format')
        sys.exit(-1)

    sys.exit(0)

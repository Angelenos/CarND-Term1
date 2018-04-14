# Udacity Autonomous Driving Nano Degree Term 1 Project 4: Advanced Lane Finding
# Created by Fengwen Song
# This is the implementation of image processing pipe line

import cv2
import sys
import pickle
import numpy as np
import matplotlib as plt
from moviepy.editor import VideoFileClip


# Definition of line class
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
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None

    def copy(self, lane):
        # was the line detected in the last iteration?
        self.detected = lane.detected
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = lane.radius_of_curvature
        # distance in meters of vehicle center from the line
        self.line_base_pos = lane.line_base_pos
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([i for i in lane.diffs], dtype='float')
        # x values for detected line pixels
        self.allx = [i for i in lane.allx]
        # y values for detected line pixels
        self.ally = [i for i in lane.ally]


# Definition of hyper parameters
cam_mtx = 0
dist_coef = 0
ksize = 5
window_width = 50
window_height = 80
lane_width = 680  # Default lane width at bottom of the image in pixels
line_width = 26  # Default line width at bottom of the image in pixels
margin = 2 * line_width
ym_per_pix = 30/720  # meters per pixel in y dimension
xm_per_pix = 3.7/700  # meters per pixel in x dimension
curv_thresh = 0.1  # Curvature threshold for lane validation
fit_thresh = 0.1  # Fit threshold for lane validation
src_pts = np.float32([(237, 690), (1043, 690), (601, 445), (678, 445)])  # Ref pts for Perspective transformation
dst_pts = np.float32([(300, 720), (980, 720), (300, 0), (980, 0)])  # Ref pts for Perspective transformation
lane_left = Line()
lane_right = Line()


def abs_sobel_thresh(img, sobel_kernel=3, orient='x', thresh=(0, 255)):
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    abs_sobel = 0
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
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
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
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
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
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
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s_channel = hls[:, :, 2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output


def lane_filter(image):
    # Create a grayscale copy of the undistorted image
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gradx = abs_sobel_thresh(gray, orient='x', sobel_kernel=ksize, thresh=(20, 200))
    grady = abs_sobel_thresh(gray, orient='y', sobel_kernel=ksize, thresh=(20, 200))
    mag_binary = mag_thresh(gray, sobel_kernel=ksize, thresh=(35, 255))
    dir_binary = dir_threshold(gray, sobel_kernel=ksize, thresh=(0.75, 1.4))
    his_binary = hls_select(image, thresh=(160, 255))
    # cv2.imwrite('output_images/debug/grad.jpg', (gradx & grady) * 255)
    # cv2.imwrite('output_images/debug/mag.jpg', mag_binary * 255)
    # cv2.imwrite('output_images/debug/dir.jpg', dir_binary * 255)
    # cv2.imwrite('output_images/debug/hls.jpg', his_binary * 255)
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (his_binary == 1)] = 1
    # cv2.imwrite('output_images/debug/filter.jpg', combined * 255)
    return combined


def slide_windows(image):
    window = np.ones(window_width)  # Create our window template that we will use for convolutions
    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template

    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(image[int(3 * image.shape[0] / 4):, :int(image.shape[1] / 2)], axis=0)
    l_center = np.argmax(np.convolve(window, l_sum)) - window_width / 2
    l_max = np.max(np.convolve(window, l_sum))
    r_sum = np.sum(image[int(3 * image.shape[0] / 4):, int(image.shape[1] / 2):], axis=0)
    r_center = np.argmax(np.convolve(window, r_sum)) - window_width / 2 + int(image.shape[1] / 2)
    r_max = np.max(np.convolve(window, r_sum))
    if l_max > 1 or r_max > 1:
        if l_max < 1:
            l_center = r_center - 680
        elif r_max < 1:
            r_center = l_center + 680
        # Add what we found for that layer
    else:
        # If both lanes can not be detected at the bottom of images, push the default value 300 and 908 (based on
        # perspective transform) into the list of window centroids
        l_center = 300
        r_center = 980

    lane_left.allx = np.array([l_center], dtype=float)
    lane_right.allx = np.array([r_center], dtype=float)
    lane_left.ally = np.array([image.shape[0]], dtype=float)
    lane_right.ally = np.array([image.shape[0]], dtype=float)

    # Go through each layer looking for max pixel locations
    for level in range(1, int(image.shape[0] / window_height)):
        # convolve the window into the vertical slice of the image
        image_layer = np.sum(
            image[int(image.shape[0] - (level + 1) * window_height):int(image.shape[0] - level * window_height), :],
            axis=0)
        conv_signal = np.convolve(window, image_layer)
        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference is at right side of window,
        # not center of window
        offset = window_width / 2
        l_min_index = int(max(l_center + offset - margin, 0))
        l_max_index = int(min(l_center + offset + margin, image.shape[1]))
        l_center = np.argmax(conv_signal[l_min_index:l_max_index]) + l_min_index - offset
        l_max = np.max(conv_signal[l_min_index:l_max_index])
        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center + offset - margin, 0))
        r_max_index = int(min(r_center + offset + margin, image.shape[1]))
        r_center = np.argmax(conv_signal[r_min_index:r_max_index]) + r_min_index - offset
        r_max = np.max(conv_signal[r_min_index:r_max_index])
        # Determine if the maximal value from left/right lanes are < 1, which indicates no lane segments found in this
        # slide of image
        if l_max > 1 or r_max > 1:
            if l_max < 1:
                l_center = r_center - (lane_right.allx[-1] - lane_left.allx[-1])
            elif r_max < 1:
                r_center = l_center + (lane_right.allx[-1] - lane_left.allx[-1])
            # Add what we found for that layer
            lane_left.allx = np.append(lane_left.allx, [float(l_center)])
            lane_right.allx = np.append(lane_right.allx, [float(r_center)])
            lane_left.ally = np.append(lane_left.ally, [float(image.shape[0] - level * window_height)])
            lane_right.ally = np.append(lane_right.ally, [float(image.shape[0] - level * window_height)])

    # If any of the lane is detected
    if len(lane_right.allx) == len(lane_left.allx) > 0:
        lane_left.detected = True
        lane_right.detected = True
        fit_left = np.polyfit(lane_left.ally * ym_per_pix, lane_left.allx * xm_per_pix, 2)
        fit_right = np.polyfit(lane_right.ally * ym_per_pix, lane_right.allx * xm_per_pix, 2)
        if len(lane_left.current_fit) == 1:
            lane_left.diffs = fit_left
            lane_right.diffs = fit_right
        else:
            lane_left.diffs = [fit_left[i] - lane_left.current_fit[i] for i in range(len(fit_left))]
            lane_right.diffs = [fit_right[i] - lane_right.current_fit[i] for i in range(len(fit_right))]
        lane_left.current_fit = fit_left
        lane_right.current_fit = fit_right
        lane_left.radius_of_curvature = \
            ((1 + (2 * lane_left.current_fit[0] * 720 * ym_per_pix + lane_left.current_fit[1]) ** 2) ** 1.5) \
            / np.absolute(2 * lane_left.current_fit[0])
        lane_right.radius_of_curvature = \
            ((1 + (2 * lane_left.current_fit[0] * 720 * ym_per_pix + lane_left.current_fit[1]) ** 2) ** 1.5) \
            / np.absolute(2 * lane_left.current_fit[0])
        lane_left.line_base_pos = (lane_left.current_fit[0] * (720 * ym_per_pix) ** 2) + \
                                  (lane_left.current_fit[1] * 720 * ym_per_pix) + \
                                  (lane_left.current_fit[2] - 300 * ym_per_pix)
        lane_right.line_base_pos = (lane_right.current_fit[0] * (720 * ym_per_pix) ** 2) + \
                                   (lane_right.current_fit[1] * 720 * ym_per_pix) + \
                                   (lane_right.current_fit[2] - 980 * ym_per_pix)


def lane_valid():
    curv_left = lane_left.radius_of_curvature
    curv_right = lane_right.radius_of_curvature
    fit_left = lane_left.current_fit
    fit_right = lane_right.current_fit
    curv_comp = np.max([np.abs(curv_left - curv_right)/curv_left, np.abs(curv_left - curv_right)/curv_right])
    fit_comp = np.max([np.abs(fit_left[0] - fit_right[0])/fit_left[0], np.abs(fit_left[0] - fit_right[0])/fit_right[0]])
    res = (curv_comp < curv_thresh) & (fit_comp < fit_thresh)
    return res


def lane_plot(shape):
    lanes = np.zeros(shape, dtype=np.uint8)
    y = np.array([i for i in range(shape[0])])
    x1 = ((lane_left.current_fit[0] * (y * ym_per_pix) ** 2) +
          (lane_left.current_fit[1] * (y * ym_per_pix)) + lane_left.current_fit[2]) / xm_per_pix
    x2 = ((lane_right.current_fit[0] * (y * ym_per_pix) ** 2) +
          (lane_right.current_fit[1] * (y * ym_per_pix)) + lane_right.current_fit[2]) / xm_per_pix
    left_pts = np.transpose(np.vstack([x1, y]))
    right_pts = np.transpose(np.vstack([x2, y]))
    box = np.int_([np.concatenate([left_pts, right_pts[::-1]])])
    left_lane = np.int_([np.concatenate([np.transpose(np.vstack([x1 - 13, y])),
                                         np.transpose(np.vstack([x1[::-1] + 13, y[::-1]]))])])
    right_lane = np.int_([np.concatenate([np.transpose(np.vstack([x2 - 13, y])),
                                         np.transpose(np.vstack([x2[::-1] + 13, y[::-1]]))])])
    # Plot green box between 2 lanes
    cv2.fillPoly(lanes, box, (0, 255, 0))
    # Plot left lane in red
    cv2.fillPoly(lanes, left_lane, (0, 0, 255))
    # Plot right lane in red
    cv2.fillPoly(lanes, right_lane, (0, 0, 255))
    return lanes


def pipeline(image):
    # This is the pipeline function used to process each image passed by main function and get information including
    # Position of lanes, curvature and vehicle position relative to lanes
    # Define 2 line objects for the left and right lanes
    # 1st stage, get undistorted images
    undist = cv2.undistort(image, cam_mtx, dist_coef, None, cam_mtx)

    # 2nd stage, filter out the possible lanes in the images
    filtered = lane_filter(undist)
    # cv2.imwrite('output_images/debug/filter.jpg', filtered*255)

    # 3rd stage, apply perspective transform to the image
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    M_inv = cv2.getPerspectiveTransform(dst_pts, src_pts)
    warped = cv2.warpPerspective(filtered, M, (filtered.shape[1], filtered.shape[0]), flags=cv2.INTER_LINEAR)
    # cv2.imwrite('output_images/debug/warped.jpg', warped * 255)

    # 4th stage, Adopt convolution method to determine details about lane
    slide_windows(warped)

    # 5th stage, compare the left and right lane to see if they are realistic
    valid = lane_valid()

    # 6th stage, draw lanes on the warped images
    lanes = lane_plot((warped.shape[0], warped.shape[1], 3))
    # cv2.imwrite('output_images/debug/lanes.jpg', lanes)

    # 7th stages, apply reverse perspective transform to get images with detected lanes and statistics plotted
    lanes_unwarped = cv2.warpPerspective(lanes, M_inv, (lanes.shape[1], lanes.shape[0]), flags=cv2.INTER_LINEAR)
    output = cv2.addWeighted(image, 1, lanes_unwarped, 0.35, 0)
    rad_curv_avg = (lane_left.radius_of_curvature + lane_right.radius_of_curvature) / 2
    loc_avg = (lane_left.line_base_pos - lane_right.line_base_pos) / 2
    # Add curvature into the output image
    output = cv2.putText(output,
                         'Curvature: {:.3f}m'.format(rad_curv_avg),
                         org=(460, 650),
                         fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                         fontScale=1,
                         color=(255, 255, 255),
                         thickness=2)
    # Add vehicle location into the output image
    output = cv2.putText(output,
                         'Vehicle location: {:.3f}m'.format(loc_avg),
                         org=(425, 600),
                         fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                         fontScale=1,
                         color=(255, 255, 255),
                         thickness=2)
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
        img_in = cv2.imread(f_input)
        if img_in is not None:
            img_out = pipeline(img_in)
            cv2.imwrite(output_file, img_out)
        else:
            print('Can''t find images')
            sys.exit(-1)
    elif '.mp4' in file_input[-1]:
        clip1 = VideoFileClip(f_input)
        lane_clip = clip1.fl_image(pipeline)
        lane_clip.write_videofile(output_file, audio=False)
    else:
        print('Not a valid input format')
        sys.exit(-1)

    sys.exit(0)

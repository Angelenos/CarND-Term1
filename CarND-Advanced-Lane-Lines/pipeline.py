################################################################################################
######## Udacity Autonomous Driving Nano Degree Term 1 Project 4: Advanced Lane Finding ########
########                             Created by Fengwen Song                            ########
################################################################################################
# This is the implementation of image processing pipe-line
# Please run the function by calling python pipeline.py + input file + output folder, here input file can be video
# in mp4 format or image in jpg format. Output folder is where the processed file will be stored. Output file will be
# Under this folder with "out_" + original file name
# Program tested on personal laptop with Intel 6820HK CPU + Nvidia GTX980 Graphic Card

# Import relevant modules
import cv2
import sys
import pickle
import numpy as np
from moviepy.editor import VideoFileClip


# Definition of line class
class Line:
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # will this frame of video/image be skipped?
        self.skip = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coeffifients of the last n fits of the line
        self.recent_fit = []
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = []
        # radius of curvature of the line in some units
        self.radius_of_curvature = 0
        # distance in meters of vehicle center from the line
        self.line_base_pos = 0
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None


# Definition of hyper parameters
cam_mtx = 0  # Pre-definition of camera matrix
dist_coef = 0  # Pre-definition of distortion coefficients
ksize = 5  # Kernel size used in sobel operator
window_width = 40  # Width of the convolution window in pixels
window_height = 80  # Window height during window slidings
line_width = 24  # Default line width at bottom of the image in pixels
margin = 8 * line_width  # Margin during window sliding to indicate the possible location of the adjacent lane segment
src_pts = np.float32([(237, 690), (1043, 690), (682, 450), (590, 450)])  # Ref pts for Perspective transformation
dst_pts = np.float32([(325, 720), (955, 720), (955, 0), (300, 0)])  # Ref pts for Perspective transformation
lane_width = dst_pts[1][0] - dst_pts[0][0]  # Default lane width at bottom of the image in pixels
ym_per_pix = 30 / dst_pts[0][1]  # meters per pixel in y dimension
xm_per_pix = 3.7 / lane_width  # meters per pixel in x dimension
curv_thresh = 3  # Curvature threshold for lane validation
fit_thresh = 6  # Threshold when compared the 2nd order polynomial coefficients with the previous fit
lane_thresh = 200  # Valid lane threshold after convolution
res_thresh = 3000  # Threshold for the residual after polynomial fit
num_it = 6  # Store result from last 6 iterations
num_layer = 9  # Default number of sliding layers
# Line objectes for left and right lanes
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
    # 1) Convert image in RGB to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    # 2) Pick the S channel
    s_channel = hls[:, :, 2]
    # 3) Make a gray scale copy of the source image
    binary_output = np.zeros_like(s_channel)
    # 4) Apply the threshold to the source image
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output


def curvature_cal(x, y):
    # Function to calculate the curvature given x and y value in pixels
    fit = np.polyfit(y * ym_per_pix, x * xm_per_pix, 2)
    curv = ((1 + (2 * fit[0] * dst_pts[0][1] * ym_per_pix + fit[1]) ** 2) ** 1.5) / np.absolute(2 * fit[0])
    return curv


def lane_filter(image):
    # Create a grayscale copy of the undistorted image
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gradx = abs_sobel_thresh(gray, orient='x', sobel_kernel=ksize, thresh=(20, 200))
    grady = abs_sobel_thresh(gray, orient='y', sobel_kernel=ksize, thresh=(20, 200))
    mag_binary = mag_thresh(gray, sobel_kernel=ksize, thresh=(60, 255))
    dir_binary = dir_threshold(gray, sobel_kernel=ksize, thresh=(0.8, 1.4))
    hls_binary = hls_select(image, thresh=(160, 255))
    combined = np.zeros_like(dir_binary)
    # Combine results from all detection results above
    # Here results from gradx, grady, mag_binary and dir_binary are calculated by operation "and" and then take "or"
    # operation with the results from "HLS_binary" to give as clean result as possible
    combined[((gradx == 1) & (grady == 1) & (mag_binary == 1) & (dir_binary == 1)) | (hls_binary == 1)] = 1
    return combined


def slide_windows(image):
    window = np.ones(window_width)  # Create our window template that we will use for convolutions
    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template
    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(image[int(4 * image.shape[0] / 5):, int(image.shape[1] / 5):int(image.shape[1] / 2)], axis=0)
    l_center = np.argmax(np.convolve(window, l_sum)) - window_width / 2 + int(image.shape[1] / 5)
    l_max = np.max(np.convolve(window, l_sum))
    r_sum = np.sum(image[int(4 * image.shape[0] / 5):, int(image.shape[1] / 2):int(4 * image.shape[1] / 5)], axis=0)
    r_center = np.argmax(np.convolve(window, r_sum)) - window_width / 2 + int(image.shape[1] / 2)
    r_max = np.max(np.convolve(window, r_sum))
    if l_max > lane_thresh or r_max > lane_thresh:
        if l_max < lane_thresh:
            l_center = int(r_center - lane_width)
        elif r_max < lane_thresh:
            r_center = int(l_center + lane_width)
        # Add what we found for that layer
    else:
        # If both lanes can not be detected at the bottom of images, push the default value 300 and 980 (based on
        # perspective transform) into the list of window centroids
        if lane_left.detected or lane_right.detected:
            l_center = int(dst_pts[0][0])
            r_center = int(dst_pts[1][0])

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
        # Expand window size when sliding over the further segments of the image
        window_width_cor = int((100 - window_width) / (image.shape[0] / window_height) * level)
        window = np.ones(window_width + window_width_cor)
        conv_signal = np.convolve(window, image_layer)
        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference is at right side of window,
        # not center of window
        offset = int(window_width / 2)
        l_min_index = int(max(l_center + offset - margin, 0))
        l_max_index = int(min(l_center + offset + margin, image.shape[1]))
        if l_min_index >= l_max_index:
            l_max_index = int(l_min_index + lane_width)
        l_center = np.argmax(conv_signal[l_min_index:l_max_index]) + l_min_index - offset
        l_max = np.max(conv_signal[l_min_index:l_max_index])
        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center + offset - margin, 0))
        r_max_index = int(min(r_center + offset + margin, len(conv_signal)))
        if r_min_index >= r_max_index:
            r_min_index = int(r_max_index - lane_width)
        r_center = np.argmax(conv_signal[r_min_index:r_max_index]) + r_min_index - offset
        r_max = np.max(conv_signal[r_min_index:r_max_index])
        # Determine if the maximal value from left/right lanes are < 1, which indicates no lane segments found in this
        # slide of image
        if (l_max > lane_thresh) or (r_max > lane_thresh):
            if l_max < lane_thresh:
                if lane_left.bestx is not None:
                    l_center = r_center - int(lane_right.bestx[level] - lane_left.bestx[level])
                else:
                    l_center = r_center - lane_width
            elif r_max < lane_thresh:
                if lane_right.bestx is not None:
                    r_center = l_center + int(lane_right.bestx[level] - lane_left.bestx[level])
                else:
                    r_center = l_center + lane_width
            elif (r_center - l_center <= lane_width * 0.85) or (r_center - l_center >= lane_width * 1.15):
                if lane_left.bestx is not None and lane_right.bestx is not None:
                    l_center_pred = (lane_left.best_fit[0] * (image.shape[0] - level * window_height) ** 2) + \
                                    (lane_left.best_fit[1] * (image.shape[0] - level * window_height)) + \
                                    (lane_left.best_fit[2])
                    r_center_pred = (lane_right.best_fit[0] * (image.shape[0] - level * window_height) ** 2) + \
                                    (lane_right.best_fit[1] * (image.shape[0] - level * window_height)) + \
                                    (lane_right.best_fit[2])
                    if np.abs(l_center_pred - l_center) > np.abs(r_center_pred - r_center):
                        l_center = int(l_center_pred)
                    else:
                        r_center = int(r_center_pred)
                else:
                    if l_max > r_max:
                        r_center = int(l_center + lane_width)
                    else:
                        l_center = int(r_center - lane_width)
            # Add what we found for that layer
            lane_left.allx = np.append(lane_left.allx, [float(l_center)])
            lane_right.allx = np.append(lane_right.allx, [float(r_center)])
            lane_left.ally = np.append(lane_left.ally, [float(image.shape[0] - level * window_height)])
            lane_right.ally = np.append(lane_right.ally, [float(image.shape[0] - level * window_height)])

    # This is the primary sanity check integrated in the window sliding
    if len(lane_right.allx) == len(lane_left.allx) == num_layer:  # If each sliding layer gives valid result
        # Mark both lanes as "detected"
        lane_left.detected = True
        lane_right.detected = True
        # Adopt full 2nd order polynomial fit. Results contains info such as coefficients and residues used to evaluate
        # the validity of fitting
        fit_left = np.polyfit(lane_left.ally, lane_left.allx, 2, full=True)
        fit_right = np.polyfit(lane_right.ally, lane_right.allx, 2, full=True)
        # If large residue was found, which indicates poor fitting results
        if (fit_left[1][0] > res_thresh or fit_right[1][0] > res_thresh) and \
                len(lane_left.current_fit) == len(lane_right.current_fit) > 0:
            # If previous valid result has been stored, marked lane as "not detected" and lane_valid() function
            # will apply fix to this result
            lane_left.detected = False
            lane_right.detected = False
        else:
            if len(lane_left.current_fit) == len(lane_right.current_fit) == 0:
                lane_left.diffs = fit_left[0]
                lane_right.diffs = fit_right[0]
            else:
                lane_left.diffs = [fit_left[0][i] - lane_left.current_fit[i] for i in range(len(fit_left[0]))]
                lane_right.diffs = [fit_right[0][i] - lane_right.current_fit[i] for i in range(len(fit_right[0]))]
            lane_left.current_fit = fit_left[0]
            lane_right.current_fit = fit_right[0]
            lane_left.radius_of_curvature = curvature_cal(lane_left.allx, lane_left.ally)
            lane_right.radius_of_curvature = curvature_cal(lane_right.allx, lane_right.ally)
    else:  # If any of the sliding layer can't give results for either lane
        # Marked lane as "not detected" and lane_valid() function will apply fix to this result
        lane_left.detected = False
        lane_right.detected = False


def lane_valid():
    # Function to check the sanity of lane detection results
    # If the lane detection result passed the primary sanity check during window sliding
    if lane_left.detected is True and lane_right.detected is True:
        curv_left = lane_left.radius_of_curvature
        curv_right = lane_right.radius_of_curvature
        curv_comp = np.max([np.abs((curv_left - curv_right)/curv_left), np.abs((curv_left - curv_right)/curv_right)])
        fit_diff = np.max([np.abs(lane_left.diffs[0] / (lane_right.diffs[0] + 1e-7)),
                          np.abs(lane_right.diffs[0] / (lane_left.diffs[0] + 1e-7))])
        res = (curv_comp < curv_thresh) & (fit_diff < fit_thresh)

        if res:  # Lane detected passes the sanity check.
            # Append the detected x values into the recent fit list
            if len(lane_left.recent_xfitted) < num_it:
                lane_left.recent_xfitted.append(lane_left.allx)
                lane_left.recent_fit.append(lane_left.current_fit)
            else:
                lane_left.recent_xfitted.pop(0)
                lane_left.recent_xfitted.append(lane_left.allx)
                lane_left.recent_fit.pop(0)
                lane_left.recent_fit.append(lane_left.current_fit)
            if len(lane_right.recent_xfitted) < num_it:
                lane_right.recent_xfitted.append(lane_right.allx)
                lane_right.recent_fit.append(lane_right.current_fit)
            else:
                lane_right.recent_xfitted.pop(0)
                lane_right.recent_xfitted.append(lane_right.allx)
                lane_right.recent_fit.pop(0)
                lane_right.recent_fit.append(lane_right.current_fit)
            # Re-calculate bestx by averaging the updated recent fit list
            lane_left.bestx = np.mean(lane_left.recent_xfitted, axis=0)
            lane_right.bestx = np.mean(lane_right.recent_xfitted, axis=0)
            # Calculate the average formula from the recent "num_it" valid detection results
            lane_left.best_fit = np.mean(lane_left.recent_fit, axis=0)
            lane_right.best_fit = np.mean(lane_right.recent_fit, axis=0)
            # Use the average formula to smooth the detected result
            lane_left.current_fit = lane_left.best_fit
            lane_right.current_fit = lane_right.best_fit
        else:  # Lane detected fails the sanity check.
            # Set boolen value of detected as False as a reminder for the next loop
            lane_left.detected = lane_right.detected = False
            # Replace the detected x value with the average one from previous valid results
            lane_left.allx = np.int_(lane_left.bestx)
            lane_right.allx = np.int_(lane_right.bestx)
            lane_left.ally = np.linspace(dst_pts[0][1], window_height, num_layer)
            lane_right.ally = np.linspace(dst_pts[0][1], window_height, num_layer)
            # Replace the polynomial fitted formula with the average one from previous valid results
            lane_left.current_fit = lane_left.best_fit
            lane_right.current_fit = lane_right.best_fit
            # Re-calculate the difference between coefficients from current fit and previous one
            lane_left.diffs = [lane_left.best_fit - lane_left.recent_fit[-1][i]
                               for i in range(len(lane_left.best_fit))]
            lane_right.diffs = [lane_right.best_fit - lane_right.recent_fit[-1][i]
                                for i in range(len(lane_right.best_fit))]
            # Re-calculate the radius of curvature
            lane_left.radius_of_curvature = curvature_cal(lane_left.allx, lane_left.ally)
            lane_right.radius_of_curvature = curvature_cal(lane_right.allx, lane_right.ally)
        # Calculate the line base position with the detected/recovered results
        lane_left.line_base_pos = ((lane_left.current_fit[0] * (dst_pts[0][1] ** 2)) +
                                   (lane_left.current_fit[1] * dst_pts[0][1]) +
                                   (lane_left.current_fit[2] - dst_pts[0][0])) * xm_per_pix
        lane_right.line_base_pos = ((lane_right.current_fit[0] * (dst_pts[0][1] ** 2)) +
                                    (lane_right.current_fit[1] * dst_pts[0][1]) +
                                    (lane_right.current_fit[2] - dst_pts[1][0])) * xm_per_pix
    else:  # If the results even fails the primary sanity check during window sliding
        # If this is not the first frame of the video/image, adopt the same method as the fail case during sanity check
        # above
        if len(lane_left.recent_fit) == len(lane_right.recent_fit) > 0:
            lane_left.allx = np.int_(lane_left.bestx)
            lane_right.allx = np.int_(lane_right.bestx)
            lane_left.ally = np.linspace(dst_pts[0][1], window_height, len(lane_left.allx))
            lane_right.ally = np.linspace(dst_pts[0][1], window_height, len(lane_right.allx))
            lane_left.current_fit = lane_left.best_fit
            lane_right.current_fit = lane_right.best_fit
            lane_left.radius_of_curvature = curvature_cal(lane_left.allx, lane_left.ally)
            lane_right.radius_of_curvature = curvature_cal(lane_right.allx, lane_right.ally)
            # Calculate the line base position with the recovered results
            lane_left.line_base_pos = ((lane_left.current_fit[0] * (dst_pts[0][1] ** 2)) +
                                       (lane_left.current_fit[1] * dst_pts[0][1]) +
                                       (lane_left.current_fit[2] - dst_pts[0][0])) * xm_per_pix
            lane_right.line_base_pos = ((lane_right.current_fit[0] * (dst_pts[0][1] ** 2)) +
                                        (lane_right.current_fit[1] * dst_pts[0][1]) +
                                        (lane_right.current_fit[2] - dst_pts[1][0])) * xm_per_pix
        else:  # If it is the first frame of the video/image, skip this frame
            lane_left.skip = True
            lane_left.skip = True


def lane_plot(shape):
    lanes = np.zeros(shape, dtype=np.uint8)
    y = np.array([i for i in range(shape[0])])
    x1 = ((lane_left.current_fit[0] * y ** 2) +
          (lane_left.current_fit[1] * y) + lane_left.current_fit[2])
    x2 = ((lane_right.current_fit[0] * y ** 2) +
          (lane_right.current_fit[1] * y) + lane_right.current_fit[2])
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
    cv2.fillPoly(lanes, left_lane, (255, 0, 0))
    # Plot right lane in red
    cv2.fillPoly(lanes, right_lane, (255, 0, 0))
    return lanes


def pipeline(image):
    # This is the pipeline function used to process each image passed by main function and get information including
    # Position of lanes, curvature and vehicle position relative to lanes
    # 1st stage, get undistorted images
    undist = cv2.undistort(image, cam_mtx, dist_coef, None, cam_mtx)

    # 2nd stage, filter out the possible lanes in the images
    filtered = lane_filter(undist)

    # 3rd stage, apply perspective transform to the image
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    M_inv = cv2.getPerspectiveTransform(dst_pts, src_pts)
    warped = cv2.warpPerspective(filtered, M, (filtered.shape[1], filtered.shape[0]), flags=cv2.INTER_LINEAR)

    # 4th stage, Adopt convolution method to determine details about lane
    slide_windows(warped)

    # 5th stage, compare the left and right lane to see if they are realistic
    lane_valid()

    # If the frame will not be skipped
    if not (lane_left.skip or lane_right.skip):
        # 6th stage, draw lanes on the warped images
        lanes = lane_plot((warped.shape[0], warped.shape[1], 3))

        # 7th stages, apply reverse perspective transform to get images with detected lanes and statistics plotted
        lanes_unwarped = cv2.warpPerspective(lanes, M_inv, (lanes.shape[1], lanes.shape[0]), flags=cv2.INTER_LINEAR)
        output = cv2.addWeighted(image, 1, lanes_unwarped, 0.35, 0)
        rad_curv_avg = (lane_left.radius_of_curvature + lane_right.radius_of_curvature) / 2
        loc_avg = (lane_left.line_base_pos + lane_right.line_base_pos) / 2
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
    else:
        output = image
        lane_left.skip = False
        lane_right.skip = False

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
        clip1 = VideoFileClip(f_input)  # .subclip(11, 14)
        lane_clip = clip1.fl_image(pipeline)
        lane_clip.write_videofile(output_file, audio=False)
    else:
        print('Not a valid input format')
        sys.exit(-1)

    sys.exit(0)

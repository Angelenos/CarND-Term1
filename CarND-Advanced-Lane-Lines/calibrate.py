# Udacity Autonomous Driving Nano Degree Term 1 Project 4: Advanced Lane Finding
# Created by Fengwen Song
# This is the code that calibrate the cameras from chessboard pictures given

import numpy as np
import pickle
import cv2
import glob
import matplotlib.pyplot as plt

nx = 9  # Number of intersection in x axis
ny = 6  # Number of intersection in y axis
shape = None
imgpoints = []
objpoints = []

# Pre-define the object points
objp = np.zeros((ny * nx, 3), np.float32)
objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

# Adopt glob function to read all calibration images
images = glob.glob('./camera_cal/calibration[1-5].jpg')
for image in images:
    # Load calibration image
    img = cv2.imread(image)
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Determine shape of the gray scaled images
    if shape is None:
        shape = gray.shape[::-1]
    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    # If corners found, add object points, image points
    if ret:
        imgpoints.append(corners)
        objpoints.append(objp)

# Determine camera matrix and distortion coefficients
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, shape, None, None)
print(mtx)
print(dist)

# Save results from undistortion
dist_res = {'Cam Mat': mtx, 'Dist Coeff': dist}
pickle.dump(dist_res, open('Distortion.p', 'wb'))

# Verify the correction on test image
test_img = cv2.imread('./camera_cal/test.jpg')
if test_img is not None:
    dst = cv2.undistort(test_img, mtx, dist, None, mtx)
    # Show images
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(test_img)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(dst)
    ax2.set_title('Undistorted Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()

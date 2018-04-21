################################################################################################
########## Udacity Autonomous Driving Nano Degree Term 1 Project 5: Vehicle Detection ##########
########                             Created by Fengwen Song                            ########
################################################################################################
# This is the implementation of vehicle detection pipeline

# Program tested on personal laptop with Intel 6820HK CPU + Nvidia GTX980 Graphic Card

import sys, os, getopt, pickle

from moviepy.editor import VideoFileClip
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from lesson_functions import *


# Definition of Hyper parameters
Ystart = 400
Ystop = 656
# Definition of feature hyper parameters
Spatial_size = (32, 32)
Hist_bins = 32
Orient = 9
Pix_per_cell = 8
Cell_per_block = 2
Hog_channel = 0
# Definition of SVC hyper parameters
epochs = 3
valid_port = 0.2
C_SVC = 1.0
gamma_SVC = 1.2

'''
def find_cars(img, ystart, ystop, scale, svc, x_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    draw_img = np.copy(img)
    img = img.astype(np.float32) / 255

    img_tosearch = img[ystart:ystop, :, :]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient * cell_per_block ** 2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = x_scaler.transform(
                np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                cv2.rectangle(draw_img, (xbox_left, ytop_draw + ystart),
                              (xbox_left + win_draw, ytop_draw + win_draw + ystart), (0, 0, 255), 6)

    return draw_img


def pipeline(image):
    return image

'''


def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # 1) Define an empty list to receive features
    img_features = []
    # 2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = np.copy(img)
    # 3) Compute spatial features if flag is set
    if spatial_feat:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        # 4) Append features to list
        img_features.append(spatial_features)
    # 5) Compute histogram features if flag is set
    if hist_feat:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        # 6) Append features to list
        img_features.append(hist_features)
    # 7) Compute HOG features if flag is set
    if hog_feat:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:, :, channel], orient, pix_per_cell,
                                                     cell_per_block, vis=False, feature_vec=True))
        else:
            hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # 8) Append features to list
        img_features.append(hog_features)

    # 9) Return concatenated array of features
    return np.concatenate(img_features)


def training(input_file):
    data = []
    y_data = []
    files = []
    paths = os.listdir(input_file + '\\vehicles')
    for path in paths:
        if '.' not in path:
            pngs = os.listdir(input_file + '\\vehicles\\' + path)
            for png in pngs:
                if 'png' in png or 'jpg' in png:
                    files.append(input_file + '\\vehicles\\' + path + '\\' + png)
                    y_data.append(1)
    paths = os.listdir(input_file + '\\non-vehicles')
    for path in paths:
        if '.' not in path:
            pngs = os.listdir(input_file + '\\non-vehicles\\' + path)
            for png in pngs:
                if 'png' in png or 'jpg' in png:
                    files.append(input_file + '\\non-vehicles\\' + path + '\\' + png)
                    y_data.append(0)

    for file in files:
        img = cv2.imread(file)
        feature = single_img_features(img, color_space='YCrCb', spatial_size=Spatial_size, hist_bins=Hist_bins,
                                      orient=Orient, pix_per_cell=Pix_per_cell, cell_per_block=Cell_per_block,
                                      hog_channel=Hog_channel, spatial_feat=True, hist_feat=True, hog_feat=True)
        data.append(feature)

    svc = SVC(C=C_SVC, kernel='linear', gamma=gamma_SVC)
    X_data = np.vstack(data).astype(np.float64)
    scaler = StandardScaler().fit(X_data)
    X_data_scale = scaler.transform(X_data)
    accuracy = []
    X_data_scale, y_data = shuffle(X_data_scale, y_data)
    for epoch in range(epochs):
        X_train, X_valid, y_train, y_valid = train_test_split(X_data_scale, y_data, test_size=valid_port, shuffle=True)
        svc.fit(X_train, y_train)
        accuracy.append(svc.score(X_valid, y_valid))
        print('{}th iteration: Accuracy is {:.3f}'.format(epoch + 1, accuracy[epoch] * 100))
    return svc


def main():
    argv = sys.argv[1:]
    try:
        opts, args = getopt.getopt(argv, "tpi:o:", ["training", "prediction", "input=", "output="])
    except getopt.GetoptError as err:
        print(err)
        sys.exit(-1)

    train = predict = False
    input_file = ''
    output_file = ''

    for o, a in opts:
        if o in ('-t', '--training'):
            train = True
            predict = False
        elif o in ('-p', '--prediction'):
            train = False
            predict = True
        elif o in ('-i', '--input'):
            input_file = a
        elif o in ('-o', '--output'):
            output_file = a
        else:
            assert False, 'Unhandled option'

    if input_file != '':
        if predict:
            f_input = input_file.split('\\')[-1]
            output_file = output_file + '\\out_' + f_input
    else:
        assert False, 'No input files'

    if train:
        print('Training mode. Training data folder: ' + input_file)
        svc = training(input_file)
        svc_model = {'model', svc}
        pickle.dump(svc_model, open(output_file, 'wb'))

    if predict:
        print('Prediction mode' + output_file)

    sys.exit(0)


if __name__ == "__main__":
    main()

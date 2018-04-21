################################################################################################
########## Udacity Autonomous Driving Nano Degree Term 1 Project 5: Vehicle Detection ##########
########                             Created by Fengwen Song                            ########
################################################################################################
# This is the implementation of vehicle detection pipeline

# Program tested on personal laptop with Intel 6820HK CPU + Nvidia GTX980 Graphic Card

import sys
import os
import getopt
import pickle
from moviepy.editor import VideoFileClip

import sklearn.utils
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

from lesson_functions import *


# Definition of Hyper parameters
Ystart = 400
Ystop = 656
thresh_heat = 4
# Definition of feature hyper parameters
Spatial_size = (32, 32)
Hist_bins = 32
Orient = 9
Pix_per_cell = 8
Pix_per_window = 64
Cell_per_block = 2
Hog_channel = 0
# Definition of SVC hyper parameters
X_scaler = None  # type: StandardScaler
svc_model = None  # type: SVC
epochs = 3  # Number of epoch in training
valid_port = 0.2  # Test size in proportion
C_SVC = 1.0  # Paramtere C in SVC
gamma_SVC = 1.2  # Parameter gamma in SVC


def find_cars(img, ystart, ystop, scale, svc, x_scaler, color_space='RGB'):
    img = img.astype(np.float32) / 255

    img_tosearch = img[ystart:ystop, :, :]
    ctrans_tosearch = np.copy(img_tosearch)
    if color_space != 'RGB':
        if color_space == 'HSV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCrCb)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // Pix_per_cell) - Cell_per_block + 1
    nyblocks = (ch1.shape[0] // Pix_per_cell) - Cell_per_block + 1

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = Pix_per_window
    nblocks_per_window = (window // Pix_per_cell) - Cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1

    # Compute individual channel HOG features for the entire image
    hog_array = [get_hog_features(ch1, Orient, Pix_per_cell, Cell_per_block, feature_vec=False),
                 get_hog_features(ch2, Orient, Pix_per_cell, Cell_per_block, feature_vec=False),
                 get_hog_features(ch3, Orient, Pix_per_cell, Cell_per_block, feature_vec=False)]

    box_list = []

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            if Hog_channel == 'ALL':
                hog_feat1 = hog_array[0][ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat2 = hog_array[1][ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat3 = hog_array[2][ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            else:
                hog_features = hog_array[Hog_channel][ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].\
                    ravel()

            xleft = xpos * Pix_per_cell
            ytop = ypos * Pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=Spatial_size)
            hist_features = color_hist(subimg, nbins=Hist_bins)

            # Scale features and make a prediction
            test_features = x_scaler.transform(
                np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale + ystart)
                win_draw = np.int(window * scale)
                box_list.append(((xbox_left, ytop_draw), (xbox_left + win_draw, ytop_draw + win_draw)))

    return box_list


def single_img_features(img, color_space='RGB', spatial_feat=True, hist_feat=True, hog_feat=True):
    # 1) Define an empty list to receive features
    img_features = []
    feature_image = []
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
        spatial_features = bin_spatial(feature_image, size=Spatial_size)
        # 4) Append features to list
        img_features.append(spatial_features)
    # 5) Compute histogram features if flag is set
    if hist_feat:
        hist_features = color_hist(feature_image, nbins=Hist_bins)
        # 6) Append features to list
        img_features.append(hist_features)
    # 7) Compute HOG features if flag is set
    if hog_feat:
        if Hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:, :, channel], Orient, Pix_per_cell,
                                                     Cell_per_block, vis=False, feature_vec=True))
        else:
            hog_features = get_hog_features(feature_image[:, :, Hog_channel], Orient,
                                            Pix_per_cell, Cell_per_block, vis=False, feature_vec=True)
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
        feature = single_img_features(img, color_space='YCrCb', spatial_feat=True, hist_feat=True, hog_feat=True)
        data.append(feature)

    svc = SVC(C=C_SVC, kernel='linear', gamma=gamma_SVC)
    X_data = np.vstack(data).astype(np.float64)
    scaler = StandardScaler().fit(X_data)
    X_data_scale = scaler.transform(X_data)
    accuracy = []
    X_data_scale, y_data = sklearn.utils.shuffle(X_data_scale, y_data)
    for epoch in range(epochs):
        X_train, X_valid, y_train, y_valid = train_test_split(X_data_scale, y_data, test_size=valid_port, shuffle=True)
        svc.fit(X_train, y_train)
        accuracy.append(svc.score(X_valid, y_valid))
        print('EPOCH {}: Accuracy is {:.3f}'.format(epoch, accuracy[epoch] * 100))

    return svc, scaler


def pipeline(image):
    scales = np.array([1, 1.25, 1.75, 2, 3, 4])
    y_start = 400
    ystop = np.int_(y_start + scales * Pix_per_window)
    bbox_list = []
    heat = np.zeros_like(image[:, :, 0]).astype(np.float)
    for i in range(len(scales)):
        bbox_list += find_cars(image, y_start, ystop[i], scales[i], svc_model, X_scaler, color_space='YCrCb')

    draw_img = np.copy(image)  # Debug
    for box in bbox_list:  # Debug
        cv2.rectangle(draw_img, box[0], box[1], (0, 0, 255), 6)  # Debug
    cv2.imwrite('output_images/debug/box.jpg', draw_img)  # Debug
    heat = add_heat(heat, bbox_list)
    heat = apply_threshold(heat, thresh_heat)
    heatmap = np.clip(heat, 0, 255)
    cv2.imwrite('output_images/debug/heatmap.jpg', heatmap)  # Debug
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(image), labels)

    return draw_img


def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "tpi:o:m:", ["training", "prediction", "input=", "output=", "Model="])
    except getopt.GetoptError as err:
        print(err)
        sys.exit(-1)

    train = predict = False
    input_file = ''
    output_file = ''
    model_file = ''
    global svc_model, X_scaler

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
        elif o in ('-m', '--model'):
            model_file = a
        else:
            assert False, 'Unhandled option'

    if input_file != '':
        if predict:
            output_file = output_file + '\\out_' + input_file.split('\\')[-1]
    else:
        assert False, 'No input files'

    if train:
        print('Training mode. Training data folder: ' + input_file)
        svc, X_scaler = training(input_file)
        svc_dict = {'model': svc, 'scaler': X_scaler}
        pickle.dump(svc_dict, open(output_file, 'wb'))
        print('Model save to ' + output_file)

    if predict:
        print('Prediction mode' + output_file)
        if model_file == '':
            files = os.listdir('.\\')
            i = 0
            while files[i][-2:] != '.p':
                i += 1
            if files[i][-2:] == '.p':
                model_file = files[i]
            else:
                print('Can''t find pickle file of trained model')
                sys.exit(-2)
        svc_dict = pickle.load(open(model_file, 'rb'))
        svc_model = svc_dict['model']
        X_scaler = svc_dict['scaler']
        if '.jpg'in input_file.split('\\')[-1] or '.png' in input_file.split('\\')[-1]:
            img_in = cv2.imread(input_file)
            if img_in is not None:
                img_out = pipeline(img_in)
                cv2.imwrite(output_file, img_out)
            else:
                print('Can''t find images')
                sys.exit(-2)
        elif '.mp4' in input_file.split('\\')[-1]:
            clip1 = VideoFileClip(input_file.split('\\')[-1]).subclip(1, 5)
            lane_clip = clip1.fl_image(pipeline)
            lane_clip.write_videofile(output_file, audio=False)
        else:
            print('Not a supported input format')
            sys.exit(-3)

    sys.exit(0)


if __name__ == "__main__":
    main()

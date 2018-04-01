# Udacity Autonomous Driving Nano-Degree Term 1 Project 3: Behavioral Cloning
# Created by Fengwen Song
# Implemented locally on MSI GT72s with CPU: Intel 6820HK, GPU: Mvidia GTX 980, RAM: 32 GB
# Edited in JetBrain PyCharm

import numpy as np
import cv2
import csv
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.models import Sequential
from keras.layers import Flatten, Dense, Cropping2D, Lambda, Convolution2D, Dropout

# Loading data from the current file
# Data corresponding to different driving strategies are stored in different folders under \data\
# Search all subfolders to acquire complete data
lines = []
images = []
angles = []
for folder in os.listdir('./data'):
    csv_folder = 'data/' + folder
    if os.path.exists(csv_folder + '/driving_log.csv'):
        with open(csv_folder + '/driving_log.csv') as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                lines.append(line)
                name_center = line[0].split('/')[-1]
                name_left = line[1].split('/')[-1]
                center_image = cv2.imread(name_center)
                if center_image is not None:
                    # Adopt images from left camera to compensate a constant negative steering angles observed when
                    # vehicle is driving on the straight line
                    left_image = cv2.imread(name_left)
                    center_angle = float(line[3])
                    if center_angle > 0:
                        left_angle = center_angle
                    else:  # Apply negative offsets, which scaled with the actual steering angles, to the original value
                        left_angle = center_angle + (0.05 * (1.8 + center_angle))
                    image_flipped = np.fliplr(center_image)
                    angle_flipped = -center_angle
                    images.append(center_image)
                    angles.append(center_angle)
                    images.append(image_flipped)
                    angles.append(angle_flipped)
                    images.append(left_image)
                    angles.append(left_angle)

# Split training and validation sample after manipulating the original ones to expand the working range
train_samples_x, validation_samples_x, train_samples_y, validation_samples_y = train_test_split(images, angles,
                                                                                                test_size=0.2)
train_size = len(train_samples_y)
valid_size = len(validation_samples_y)
print('Training set size: {}'.format(train_size))
print('Validation set size: {}'.format(valid_size))


# Definition of Generator functions
def generator(x, y, batch_size=64):
    num_samples = len(y)
    while 1:  # Loop forever so the generator never terminates
        # shuffle(x, y)
        for offset in range(0, num_samples, batch_size):
            batch_samples_x = x[offset:offset + batch_size]
            batch_samples_y = y[offset:offset + batch_size]
            # trim image to only see section with road
            x_train = np.array(batch_samples_x)
            y_train = np.array(batch_samples_y)
            yield shuffle(x_train, y_train)


# compile and train the model using the generator function
train_generator = generator(train_samples_x, train_samples_y, batch_size=128)
validation_generator = generator(validation_samples_x, validation_samples_y, batch_size=128)

# Definition of Hyper Parameters
epochs = 4
ch, row, col = 3, 65, 320  # Trimmed image format
drop_rate = 0.2

# Layout of model
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(drop_rate))
model.add(Dense(50))
model.add(Dropout(drop_rate))
model.add(Dense(10))
model.add(Dropout(drop_rate))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch=train_size,
                                     validation_data=validation_generator,
                                     nb_val_samples=valid_size, nb_epoch=epochs)

# Save Model
model.save('model.h5')

# Plot loss versus EPOCHS
plt.plot([i for i in range(1, epochs + 1)], history_object.history['loss'])
plt.plot([i for i in range(1, epochs + 1)], history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

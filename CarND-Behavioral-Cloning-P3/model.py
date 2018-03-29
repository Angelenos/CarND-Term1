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
for folder in os.listdir('./data'):
    csv_folder = 'data/' + folder
    if os.path.exists(csv_folder + '/driving_log.csv'):
        with open(csv_folder + '/driving_log.csv') as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                lines.append(line)


train_samples, validation_samples = train_test_split(lines, test_size=0.2)


# Definition of Generator functions
def generator(samples, batch_size=64):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name_center = batch_sample[0].split('/')[-1]
                name_left = batch_sample[1].split('/')[-1]
                center_image = cv2.imread(name_center)
                if center_image is not None:
                    # Adopt images from left camera to compensate a constant negative steering angles observed when
                    # vehicle is driving on the straight line
                    left_image = cv2.imread(name_left)
                    center_angle = float(batch_sample[3])
                    if center_angle > 0:
                        left_angle = center_angle
                    else:  # Apply negative offsets, which scaled with the actual steering angles, to the original value
                        left_angle = center_angle + (0.05 * (1.5 + center_angle))
                    image_flipped = np.fliplr(center_image)
                    angle_flipped = -center_angle
                    images.append(center_image)
                    angles.append(center_angle)
                    images.append(image_flipped)
                    angles.append(angle_flipped)
                    images.append(left_image)
                    angles.append(left_angle)

            # trim image to only see section with road
            x_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(x_train, y_train)


# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=64)
validation_generator = generator(validation_samples, batch_size=64)

# Definition of Hyper Parameters
epochs = 4
ch, row, col = 3, 65, 320  # Trimmed image format

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
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples),
                                     validation_data=validation_generator,
                                     nb_val_samples=len(validation_samples), nb_epoch=epochs)

# Save Model
model.save('model.h5')

# Plot loss versus EPOCHS
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

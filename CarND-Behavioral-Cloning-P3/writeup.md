# **Behavioral Cloning** 


**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/model.png "Model Visualization"
[image2]: ./examples/center.png "Center Driving"
[image3]: ./examples/edge.png "Recovery Image"
[image4]: ./examples/edge2.png "Recovery Image 2"
[image5]: ./examples/loss.png "Validation loss"
[image8]: ./examples/test.png "Test cases"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following [files](https://github.com/Angelenos/CarND-Term1/blob/master/CarND-Behavioral-Cloning-P3/):
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md or writeup_report.pdf summarizing the results
* run.mp4 the video of model simulation

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The model adopted in this project is published by Nvidia. Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 29). 20% of the training set was randomly chosen to be the validation set.

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 92).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of multiple driving patterns include:

1. Keep vehicle in the center
2. Steering right back to center when vehicle is at left edge
3. Steering left back to center when vehicle is at right edge
4. Smooth driving, which emulates the race driving patterns (passing a corner following the "outer-inner-outer" pattern)

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to adopt the Nvidia autonomous driving model and tune it with designtaed training data.

My first step was to use a convolution neural network model similar to the Nvidia network I thought this model might be appropriate because it has been proved by the Nvidia self-driving car team that this model is capable to achieve satisfactory self-driving task.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the epochs from the initial value of 5 to 3, and the validation loss is now dropping monotonously. I also planned to introduce dropout layer between fully connected ones to further reduce overfitting. However the model started to ignore some training data I include such as sharp steering and maintaining vehicle orientations when entering/exiting bridges. This indicated that with a small size of training data it is not beneficial to introduce dropout layer. Due to the some reason I reduce the size of validation set to be 20% of the entire training data to reduce the risk of not picking any of the cases I include in the training set.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track:

1. Sharp corner which requires steering angles greater than 5 degrees.
2. Corners with unpaved branches
3. The only right sharp corner close to the end
4. Entering or exiting the bridge

to improve the driving behavior in these cases, I create seperate test cases for each of this scenarios, which will be covered later.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture shares the same acchitecture with the Nvidia self-driving car model excepting adding a 2D cropping layers to remove useless information including background and sky from the training data. The architecture consists of (model.py lines 79-95):

1. Lambda layer which normalizes the original image into values between -0.5 to 0.5
2. A 2D Cropping layer which crops the upper 70 and lower 25 pixels of the image
3. 2D Conv layer with 24 filters of 5x5 and "ReLu" activation function
4. 2D Conv layer with 36 filters of 5x5 and "ReLu" activation function
5. 2D Conv layer with 48 filters of 5x5 and "ReLu" activation function
6. 2D Conv layer with 64 filters of 3x3 and "ReLu" activation function
7. 2D Conv layer with 64 filters of 3x3 and "ReLu" activation function
8. Flatten layer
9. Fully connected layer leads to 100 neurons
10. Fully connected layer leads to 50 neurons
11. Fully connected layer leads to 10 neurons
12. Fully connected layer leads to 1 neurons

#### 3. Creation of the Training Set & Training Process

![alt text][image8]

For examples, the bridge folders contains training data when vehicle driving onto/away from the bridge in different angles and relative positions on the track. Edge 1 and 2 deal with steering back to center at corners when vehicle is slightly off the center or away from center, respectively.

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to steering back to the center of the lanes when driving at the edges. These images show what a recovery looks like

![alt text][image3]

Also I include situations when the road shoulder has marks of white-red strips, which indicate sharp corners. Additional test cases are included to force the model apply higher priority when dealing with this scenarios to avoid understeering.

![alt text][image4]

To augment the data sat, I also flipped images and angles thinking that this would provide training data for right steering, which is much less than the left steering on the test track.

After the collection process, I had 10683 numbers of data points. I then preprocessed this data by normalizing the R, G and B channels with 255 and then centered to 0. Also since it is not necessary to include background such as mountains, skies, etc., all images are cropped to remove the top 70 and bottom 25 pixels. This also reduce the computation time to train the model

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by looking into the validation loss against multiple choices of epochs. It turns out 3 is the maximal value to prevent overfitting.

![alt text][image5]

I used an adam optimizer so that manually training the learning rate wasn't necessary.

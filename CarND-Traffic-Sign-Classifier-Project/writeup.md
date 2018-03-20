# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/brightness.png "Brighten"
[image3]: ./examples/rotation.png "Rotation"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 69598
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to adjust the brightness of the picture so details including color, edges and shapes can be retrieved more clearly from the training data. Here the average of R, G and B channels for each pixel are considered with same weighs and averaged to get the global brightness. If the brightness is less than 64, each channel of the entire image will be multiplied by (127 / global brightness). Here the values of 64 and 127 are chosen to avoid channel values go above 255 after manipulation

![alt text][image2]

I decided to generate additional data because traffic signs are taken with random angles from the training dataset. Therefore it will be beneficial to include additional data acquired by rotating the original dataset with a random angle (from -15 to 15 degrees) so that the model is able to minize the effect of orientation of the entire sign.

Method of image offsetting is not considered here since the training dataset that is cropped from driving videos are already adjusted to make sure the ROI is at the middle of the picture.

To add more data to the the data set, I made a copy of the original dataset, and rotate each image with a randomly generated angle between -15 to 15 degrees, and append the augmented dataset at the end of the original one before the training dataset is shuffled.

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Average pooling	      	| 2x2 stride,  outputs 14x14x6				|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 10x10x16    |
| RELU					|												|
| Average pooling	      	| 2x2 stride,  outputs 5x5x16				|
| Fully connected		| Input = 400, Output = 120		|
| RELU					|												|
| Dropout					|			keep probability = 0.75			|
| Fully connected		| Input = 120, Output = 84		|
| RELU					|												|
| Dropout					|			keep probability = 0.75			|
| Fully connected		| Input = 84, Output = 43		|
| Softmax				| |
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an batch size of 256 to balance the learning speed and memory size. Learning rate was chosen to be 0.001 based on the fact that Adam optimizer will automatically optimize learning rate and 0.001 is the recommending value by the author of this model

An EPOCH of 25 was chosen instead of 10 by default, which is due to introducing dropout and L2 regularization slows down the speed of convergence. Also since Adam optimizer is able to adjust the learning rate based on the accuracy, it will be also beneficial to extend the training cycles to ensure better final accuracy

Dropout layer was introduced into the model and a keep probability of 0.7 was chosen by trial and error to balance the learning speed and effect of preventing over-fitting

L2 Regularization was also introduced for each layer in the fully connected network and the coefficient of 0.0001 was chosen by trying value from 0.00001 to 0.01.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

In this project LeNet ConvNet model is selected.

Rationality for selecting LeNet is based on its original application: Character recognition from small images. LeNet has been proved to ensure both simplicity and accuracy in identifing limited number of patterns within a smail image, which is exactly the case for traffic sign identification. The only concern is that traffic sign contains color information, and under certain circumstances color will be essential for traffic sign identification. However from the result it is proved that the original LeNet is able to achieve high accuracy even when dealing with higher complexity like color information. On the other hand, it is also believed that introducing more layers in the convolution layer or full connection layers may potentially increase the overall accuracy further.



### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first images might be difficult to classify because it is fairly similar with "End of speed limit 80". The model is required to clearly identify and assign significant weighs to the red bar crossing the number 80 to distinguish one from the other.

The second image has a human figure in the middle, and the model also needs to distinguish that one from a deer to correctly recognize this one from the "Wild animals crossing"

The third and forth ones both have symmetric signs corresponding in the training set. These test cases will verify if the model can recognize difference from symmetry. Also the dataset contains multiple signs with left/right arrow. The model is also required to identify each from others by patterns including shape, color and arrow types

The fifth one is in triangle shape with no figure in the middle. This is similar with "No Vehicles"except the shape.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image					        |     Prediction				| 
|:-----------------------------:|:-----------------------------:| 
| Speed limit (80km/h)			| Speed limit (80km/h)			| 
| Pedestrians					| Pedestrians					|
| Dangerous curve to the left	| Dangerous curve to the left	|
| Turn right ahead				| Turn right ahead				|
| Yield							| Yield							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is fairly sure that this is a Speed limit (80km/h) (probability of 0.996), and the image does contain a Speed limit (80km/h). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .9956        			| Speed limit (80km/h)							| 
| .0037    				| Speed limit (60km/h)							|
| .0005					| Speed limit (50km/h)							|
| .0002					| Speed limit (100km/h)		 					|
| .0001					| Speed limit (30km/h)							|

For the second image, the model is fairly sure that this is a Pedestrians (probability of 0.999), and the image does contain a Pedestrians. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .9999					| Pedestrians 									| 
| <.0001				| General caution								|
| <.0001				| Right-of-way at the next intersection			|
| <.0001				| Road narrows on the right		 				|
| <.0001				| Dangerous curve to the left					|

For the third image, the model is fairly sure that this is a Dangerous curve to the left (probability of 0.995), and the image does contain a Dangerous curve to the left. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .9995					| Dangerous curve to the left					| 
| .0004					| Slippery road									|
| .0001					| Double curve									|
| <.0001				| Wild animals crossing			 				|
| <.0001				| Road narrows on the right						|

For the forth image, the model is fairly sure that this is a Turn right ahead (probability of 0.998), and the image does contain a Turn right ahead. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .9980					| Turn right ahead								| 
| .0018					| Keep left										|
| .0001					| Ahead only									|
| <.0001				| Turn left ahead								|
| <.0001				| Keep right									|

For the fifth image, the model is fairly sure that this is a Yield (probability of around 1.00), and the image does contain a Yield. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Yield											| 
| <.0001				| Priority road									|
| <.0001				| Speed limit (50km/h)							|
| <.0001				| Road work										|
| <.0001				| No passing									|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?



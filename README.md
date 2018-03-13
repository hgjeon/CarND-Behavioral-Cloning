# **Behavioral Cloning**

---

** Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/model_lenet.png "LeNet Model Visualization"
[image2]: ./images/model_nvidia.png "Final Model Visualization"
[image3]: ./images/loss_lenet.png "Loss Plot - Lenet"
[image4]: ./images/loss_nvidia_1.png "Loss Plot - 1st Learning"
[image5]: ./images/loss_nvidia_2.png "Loss Plot - 2nd Learning"

[image6]: ./images/image_normal.jpg "Normal Image"
[image7]: ./images/image_flipped.jpg "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5, and 3x3 filter sizes and depths between 24 and 64 (model.py)

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py).

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an 'adam optimizer', so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, and recovering from the left and right sides of the road.

For details about how I created the training data, see the next section.

### Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to evaluate training and validation errors. After that actual test running simulator for final verification.

My first step was to use a convolution neural network model similar to the LeNet model. I thought this model might be appropriate because it was good for the image classification application.

![alt text][image1]

In order to gauge how well the model was working, I split my image and steering angle data into a training (80%) and validation set (20%). I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. Also, validation loss is oscillating, while training loss continue going down. This implied that the model was overfitting.

![alt text][image3]

To combat the overfitting, I modified the model such as,
- Every layer includes dropout of 20%
- More training Data

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track.
- Several spots that do not have side line
- Several spots that have sharp curve
- Bridge area which has quite different view

Several approaches have been tried to improve the driving behavior in these cases.
- Preprocessing
    - Cropping input image to eliminate irrelevant features
- Additional Training Data focusing on the failure spots
    - This seemed to be an important role to improve the driving behavior
    - Training data with Center-driving
    - Training data focusing on the failure spots. Drive back and forth multiple times.
- Data Augmentation
    - Flipped image from right to left to get double size of training data
- Save/Load model
    - Since this training takes longer time, and it requires many repeat, one of the most important tactics is that save the model after one learning session, then tried different Training data sets, and parameters to see the improvement, then applied this trained weights for the next step.

At the end of the process, the vehicle is able to drive autonomously around the track multiple times without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py) is based on 'Nvidia' model, and it is consisted of a convolution neural network with the following layers and layer sizes.

Here is a visualization of the architecture.

![alt text][image2]

#### 3. Creation of the Training Set & Training Process

##### Image Preprocessing

To augment the data sat, I flipped images from left to right and at the same time change to angles multiplying (-1). In this way, data set became double.

For example, here is an image that has then been flipped:

![alt text][image6]

![alt text][image7]

After this data augmentation, total data I collected and used is summarized here:
- Default data:
    - 8036 sets --> 16072 sets
- More collection from failure analysis:
    - 8269 sets --> 16538 sets

This image has been clopped in the 1st stage of model, in which both training data, and test data will be processed in the same way.

Finally, all data is randomly shuffled the data set and put 20% of the data into a validation set.




##### First Training
For the initial training I used default training data which has multiple laps of center lane driving. After the first training with epoch size of 10, driving seems smooth, but it cannot finish the track successfully.
- Epoch 10; Batch size 64;
- Driving is smooth in general
- Most of time it stuck around bridge, or sharp curve areas

![alt text][image4]

##### Failure analysis

For the second training using the saved model from the first training, I generated much more training data focusing on those specific spots.

- Drive back and forth multiple times in those specific area
- Total 8269 data set {center images, steering angle}

##### Second training

In this way, I could make it successfully drive route 1 without huge number of data collection. Also, this seems to effectively eliminate bias from other area such as straight road most of time.

- Drive smoothly
- Drive many rounds of tracks successfully
- It was a little bit off-center at one sharp curve area, but it was still running with a bug, in which training is done with BGR image, while driving is done with RGB image.
- After fixing the bug, it could drive in center most of time. 

The second training result on top of the first trained model is shown below.

![alt text][image5]

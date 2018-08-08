# **Behavioral Cloning**

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/modelarchitecture.png "Model Visualization"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* modely.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* README.md
* run2.mp4 showing a video a successful autonomous run

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```
It should be noted that the model does not always successfully allow the car to navigate the track. Multiple attempts may be necessary for the car to successfully navigate the track using model.h5

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with three 5x5 and one 3x3 sized filters and depths between 3 and 48 (model.py lines 87-95)

The model includes ELU layers to introduce nonlinearity. The ELU layers replaced Relu layers in previous attempts in order to preserve the effects of negative scores (code lines 100,103,106,109).



#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 99,102,105,108).

The model was trained on different data sets to ensure that the model was not overfitting (code line 10-16). Random image generators and augmenters also helped prevent overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 112).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used center lane driving primarily, but also used recovery driving to help the model in places where it struggled.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to find an already existing model that was being used for a similar purpose.

My first step was to use a convolution neural network model similar to the NVIDIA self-driving car CNN. While this model operates with data obtained from actual driving scenarios, its application is almost identical to the one needed for this project.

In order to gauge how well the model was working, I would validate my results on the track. While splitting data in testing and validation sets is valuable, perhaps the best indicator of success is how the car navigates the track itself.

Initially, I trained my models for few epochs because of time constraints and the belief the a low MSE for the training data meant more epochs would cause overfitting. However, after training for 50+ epochs on different models I saw that I had been underfitting the data.

Various changes, such as changing the activation function, adding dropout layers, changing the number of layers and neurons in them, were made. However, time and time again data augmentation and increased training time proved to be the two most important factors in the models performance.

The final model weaves due to low angle data being removed from the training set, but is still able to navigate the track completely without leaving the barriers.

#### 2. Final Model Architecture

The final model architecture (model.py lines 87-113) consisted of a convolution neural network with a cropping layer, four convolutional layers with 5x5 and 3x3 filters and accompanying relu activation functions, and four normal layers with 800,100,50, and 1 neurons respectively. Each layer except for the final had an elu activation function. This model is based on the NVIDIA CNN Architecture with some minor modifications.

Here is a visualization of the NVIDIA architecture below:


![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded five laps on track one using center lane driving.

I then recorded the vehicle recovering from the left side and right sides of the road at turns where the model struggled. I would drive these areas of the track ~5 times, hugging either the right or left side or driving in the middle, to allow robust navigation of the specified turn.

After the collection process, I had 78378 data points.


To augment the data set, I also flipped images and angles in order to avoid the inevitable left turn bias on the first training course.

This data source was randomly shuffled in a generator that returned batches of 32 over 600 times an epoch. This meant that about 5 epochs would be needed to cover the data. Due to the randomness of image section however some images may never have been used in training, although after 75 epochs the odds of this are very small.

In regards to the number of epochs, it was determined that 150 epochs would cause overfitting, even when using dropout layers. After further experimentation I concluded that 75 epochs proved to be the more optimal number for training with my image generator.
# Project3

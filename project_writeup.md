# **Behavioral Cloning** 

## Jeyte Hagaley's Write up!


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report



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

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network following  [NVIDIA's](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) end to end learning architecture for self-driving cars. I followed this model very closely using activation functions I read about on my own to fine tune my model.

Below is a simplified view of my model:

* Input layer to standardize image.
* Cropping layer to use bottom half of image.
* 4 Convolution layers with RELU activations
* 3 Flatten layers with both RELU and ELU activations.
* Output layer with steering angle.

#### 2. Attempts to reduce overfitting in the model

The model contains both RELU and ELU activations to reduce overfitting and produce a better model. The model was trained, validated, and tested on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving using the center camera and recovering from the left and right sides of the road using the cameras on the side of the car using a delta of 0.2 to steer the car to try and position it in the middle of the lane. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to look at the activation functions I was using and compare the results of training with the scores of both my training and validation set. If either was too high I knew I needed to add different activations because the model was overfitting. Once I built a model were the accuracy of both the training and validation sets were low and similar I tested it against the simulator. I did this until I was able to successfully go around the whole track without any problems!

#### 2. Final Model Architecture

The final model architecture is detailed above but works very well with the designed track. My model was able to steer the car successfully without any problems. 

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I used the driving data provided to us for this project. This data was very good and was enough for my model to learn from. Once I took my data I pre processed the images.I finally randomly shuffled the data set and put 15% of the data into a test set. From the remaining data, I used Kera's `fit` function and allocated 20 percent of the data for validation. After training and validating my model I was able to get a testing loss of 0.0163553705767, which was very low! 

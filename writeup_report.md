# **Behavioral Cloning** 
---
The goal of this project for the Udacity Self-Driving Car Nanodegree is to train a deep CNN that can clone the human driving behavior in a simulator developed by Udacity. The CNN can autonomously drive the car around the track. This networks is trained by the images and steering angles that were recorded while a human was driving a car in the simulator. Thus the networks can steer the car in the simulator.

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_img/nVidia_model.png "NVIDIA Model"
[image2]: ./writeup_img/model.png "Model"
[image3]: ./writeup_img/result.jpg "Training_result"

## Rubric Points
 Here is the [rubric points](https://review.udacity.com/#!/rubrics/432/view) of this project.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* `model.py` containing the script to create and train the model
* `drive.py` for driving the car in autonomous mode
* `model.h5` containing a trained convolution neural network 
* `writeup_report.md` summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my `drive.py` file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The `model.py` file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

There are some successful CNNs architectures that have been used in cloning driving behaviors, e.g. [NVIDIA model](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) and [comma.ai model](https://github.com/commaai/research/blob/master/train_steering_model.py). Both these CNNs architectures just predict steering angle as the only output. In this project, my model is based on NVIDIA model showing as follows.

![nVidia_model][image1]
 
My model is summarized as follows:
```sh
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param       Connected to                     
____________________________________________________________________________________________________
lambda_1 (Lambda)                (None, 66, 200, 3)    0           lambda_input_1[0][0] 
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 31, 98, 24)    1824        lambda_1[0][0]                   
____________________________________________________________________________________________________
prelu_1 (PReLU)                  (None, 31, 98, 24)    72912       convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 14, 47, 36)    21636       prelu_1[0][0]                    
____________________________________________________________________________________________________
prelu_2 (PReLU)                  (None, 14, 47, 36)    23688       convolution2d_2[0][0]            
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 5, 22, 48)     43248       prelu_2[0][0]                    
____________________________________________________________________________________________________
prelu_3 (PReLU)                  (None, 5, 22, 48)     5280        convolution2d_3[0][0]            
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 5, 22, 64)     27712       prelu_3[0][0]                    
____________________________________________________________________________________________________
prelu_4 (PReLU)                  (None, 5, 22, 64)     7040        convolution2d_4[0][0]            
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 5, 22, 64)     36928       prelu_4[0][0]                    
____________________________________________________________________________________________________
prelu_5 (PReLU)                  (None, 5, 22, 64)     7040        convolution2d_5[0][0]            
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 7040)          0           prelu_5[0][0]                    
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           704100      flatten_1[0][0]                  
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 100)           0           dense_1[0][0]                    
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        dropout_1[0][0]                  
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 50)            0           dense_2[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         dropout_2[0][0]                  
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 10)            0           dense_3[0][0]                    
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          dropout_3[0][0]                  
____________________________________________________________________________________________________
Total params: 956,979
```

Here are some details need to pay attention:
* The input data of the networks need to be preprocessed (implement in `drive.py` function `preprocess_image()`): 
    * Crop the input images: (160,320,3) ==> (90,320,3)
    * Resize to small shape: (90,320,3) ==> (66,200,3)
* The first layer is Lambda layer to normalize the data:
```sh
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(66, 200, 3))
```


#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers after fully connected layers in order to reduce overfitting. 

The model was trained and validated on different data sets to ensure that the model was not overfitting. My sample data was split into training and validation data, using 80% as training data and 20% as validation data. 

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (`model.py` line 111).

### Model Architecture and Training Strategy

#### 1. Solution Design Approach
My first network model is based on NVIDIA model. Since there is not any dropouts described in NVIDIA original paper, I just tried a model without dropouts. However, I found that my model had a low mean squared error on the training set but a high mean squared error on the validation set. Obviously it is because of overfitting.

In order to overcome overfitting, I added dropout layers after fully connected layers. At last, the result is satisfactory.

#### 2. Final Model Architecture

Here is a visualization of the final model architecture:

![Model][image2]

#### 3. Creation of the Training Set & Training Process
 In my views, for this project appropriate training data are more important than model architecture. Here is my constitution of my original training data:

* One lap of center lane driving
* One lap of center lane driving counterclockwise
* One lap of recovery driving from the sides
* One lap focusing on driving smoothly around curves

In addition, I also augmented data by flipping images and taking the opposite sign of the steering measurement (`model.py`). Finally there are total 12657 data. 10125 for training and 2532 for validation.

All these data was used for training the model with 5 epochs. The data was shuffled randomly. The following picture shows the training process:

![Training_result][image3]

####4. Result 
After training the model, the car is able to autonomously drive around the first track. See this video `video.mp4`.
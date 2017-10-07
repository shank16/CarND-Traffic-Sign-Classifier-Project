# Traffic Sign Recognition 



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

[image1]: ./examples/previsualization.png "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"
[image9]: ./examples/augmented_image.png " Image Transformtion exampel"
[image10]: ./add_pics/1sh.png 
[image11]: ./add_pics/2sh.png
[image12]: ./add_pics/3sh.png 
[image13]: ./add_pics/4sh.jpg 
[image14]: ./add_pics/5sh.png 
[image15]: ./add_pics/6sh.png 
[image16]: ./add_pics/7sh.jpg 
[image17]: ./add_pics/8sh.png 
[image18]: ./add_pics/9sh.png 
[image19]: ./add_pics/10sh.png 
[image20]: ./add_pics/11sh.jpg 
[image21]: ./add_pics/12sh.jpeg 
[image22]: ./examples/softmax_probability.png 
[image23]: ./examples/bar_softmax.png 
[image24]: ./examples/convolution_first_layer.png 
[image25]: ./examples/convolution_second_layer.png 

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. 

You're reading it! and here is a link to my [project code](https://github.com/shank16/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set befor data augmentation is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)[x,y,axis]
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to do data augmentation process where I applied random brightness, rotation, translation, affine function etc. and then the data has been added to the classes where there were less than 1000 data with the data tranformation process, described earlier.

Here is the example of data augmentation process:
![alt text][image9]

After that I applied grayscaling 
Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because having hign mean creates probelm of unstable gradient descent algorithm.

 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 normalized gray scale image   							| 
| Convolution 1     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution  2   |1x1 stride, valid padding, outputs 10x10x16 |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Convolution 	 3   |1x1 stride, valid padding, outputs 1x1x400 |
| RELU					|												|
| Flatten convolution 2					|Input- 5x5x16 output-400				|
|  Add Flatten convolution 2	and convolution 3				|output-800			|
|  Dropout 				| probability - 0.5		|
| Fully connected		| Input- 800 output-43     									|

 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam optimizer with batch size of 100 , number of epoches 50 and learning rate as 0.0009. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 0.958
* test set accuracy of  0.943

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
I first chose normal LeNet model I used in previous assignment and then I added a drop out layer to it.There was a minor        change in the performance (0.933 to 0.941 ), so I changed the model to LeNet2 in the code. It was derived from  Sermanet/LeCun model used specially for the traffic sign classification.

* What were some problems with the initial architecture?
The accuracy was low. Not sure why . Having little background in CNN it is hard to judge the exact reason for its performance.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
There was not much change in the structure just added an extra convolution layer and added it to the flatten layer of the previous convolution layer. There was not much change in the performace (0.945) to (0.958), so not sure about the reason. Might be overfitting as training set accuracy was quite high. 

* Which parameters were tuned? How were they adjusted and why?
 Learning rate and batch size ,it was trial and error according to performance. 

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
As discussed in lecture videos dropout layer makes the prediction more uncertain which helps CNN to improve its performance by relying more on the available data.

If a well known architecture was chosen:
* What architecture was chosen? 
  Sermanet/LeCun
* Why did you believe it would be relevant to the traffic sign application?
  It was used before by other people for traffic sign detection. Will look into the reason in more details in future.
 

### Test a Model on New Images

#### 1. Choose twelve German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are eleven German traffic signs that I found on the web:

![alt text][image10] ![alt text][image11] ![alt text][image12] 
![alt text][image13] ![alt text][image14] ![alt text][image15]
![alt text][image16] ![alt text][image17] ![alt text][image18] 
![alt text][image19] ![alt text][image21]  


The first image might be difficult to classify because of its lower pixel value

2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

![alt text][image22]

The model was able to correctly guess 9 of the 12 traffic signs, which gives an accuracy of 75%. This compares favorably to the accuracy on the test set.

#### 3. Describe how certain the model is when predicting on each of the twelve new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

 Find the following figure for visualizing softmax probabilities for new tested images
 
![alt text][image23]
### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

My visualization is not same as the suggested in file. Have to work on that to improve it. Any suggestions are welcome. Kindly find the attched figures for visalization of first two layers.

![first layer][image24]


![second layer][image25]


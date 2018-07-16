## MIT 6.S191 - Deep Learning - Course notes 

This is a series of personal notes about the MIT 6.S191 Deep Learning courses that I have virtually attended during the last few weeks. 

[Here is a link to the course site](http://introtodeeplearning.com/).

I will go through the first four courses: 
1.  Introduction to Deep Learning 
2.  Sequence Modeling with Neural Networks 
3.  Deep learning for computer vision - Convolutional Neural Networks 
4.  Deep generative modeling 

Starting from the second course, I will add an application on an open-source dataset for each course. 

# Introduction to Deep Learning 

Traditional machine learning models have always been very powerful to handle structured data and have been widely used by businesses for credit scoring, churn prediction, consumer targeting, and so on. 

The sucess of these model highly depend on the performance of the feature engineering phase: the more we work close to the business to extract relevant knowledge from the structured data, the more powerful the model will be. 

When it comes to unstructured data (images, text, voice, videos), hand engineered features are time consuming, brittle and not scalable in practice. That is why Neural Networks become more and more popular thanks to their ability to automatically discover the representations needed for feature detection or classification from raw data. This replaces manual feature engineering and allows a machine to both learn the features and use them to perform a specific task.

Improvements in Hardware (GPUs) and Software (advanced models / research related to AI) also contributed to **deepen the learning** from data using Neural Networks.  

The fundamental bulding block of Deep Learning is **The Perceptron** which is a single neuron in a Neural Network. 

Given a finite set of _m_ inputs (e.g. _m_ words or _m_ pixels), we multiply each input by a weight (_theta 1_ to _theta m_) then we sum up the weighted combination of inputs, add a bias and finally pass them through a non-linear activation function. That produces the output _Yhat_. 

![Branching](https://raw.githubusercontent.com/ZiedHY/ZiedHY.github.io/ZiedHY-patch-1/Perceptron.PNG)

*   The bias _theta 0_ allows to add another dimension to the input space. Thus, the activation function still provide an output in case of an input vector of all zeros. It is somehow the part of the output that is independent of the input.
*   The purpose of activation functions is to introduce non-linearities into the network. In fact, linear activation functions produce linear decisions no matter the input distribution. Non-linearities allow us to better approximate arbitrarily complex functions. Here some examples of common activation functions: 
![Octocat](https://raw.githubusercontent.com/ZiedHY/ZiedHY.github.io/ZiedHY-patch-1/ActivationFunctions.PNG)

Deep Neural Networks are no more than a **stacking** of multiple perceptrons (hidden layers) to produce an output. 
![Branching](https://raw.githubusercontent.com/ZiedHY/ZiedHY.github.io/ZiedHY-patch-1/FullyConnected.PNG)

Now, once we have understood the basic architecture of a deep neural network, let us find out how it can be used for a given task. 

Let us say, for a set of X-ray images, we need the model to automatically distinguish those that are related to a sick patient from the others. 

For that, machine learning models, like humans, need to learn to differentiate between the two categories of images by **observing** some images of both sick and healthy individuals. Hence, they automatically understand patterns that better describe each category. This is what we call **the training phase**.  

Concretely, a pattern is a weighted combination of some inputs (images, parts of images or other patterns). Hence, **the training phase is nothing more than the phase during which we estimate the weights (also called parameters) of the model.** 

When we talk about estimation, we talk about an **objective function** we have to optimize. This function shall be constructed to reflect the best the performance of the training phase. When it comes to prediction tasks, this objective function is usually called **loss function** and measures the cost incurred from incorrect predictions. When the model predict something that is very close to the true output then the loss function is very low, and vice-versa. 

In the presence of input data, we calculate an empirical loss (binary cross entropy loss in case of classification and mean squared error loss in case of regression) that measures the total loss over our entire dataset: 
![Octocat](https://raw.githubusercontent.com/ZiedHY/ZiedHY.github.io/ZiedHY-patch-1/EmpiricalLossFunction.PNG)

Since the loss is a function of the network weights, our task it to find the set of weights _theta_ that achieve the lowest loss: 

![Octocat](https://raw.githubusercontent.com/ZiedHY/ZiedHY.github.io/ZiedHY-patch-1/WeightEstimation.PNG)

If we only have two weights _theta 0_ and _theta 1_, we can plot the following diagram of the loss function. What we want to do is to find the minimum of this loss and consequently the value of the weights where the loss attains its minimum. 

![Octocat](https://raw.githubusercontent.com/ZiedHY/ZiedHY.github.io/ZiedHY-patch-1/GradientDescent.PNG)

To minimize the loss function, we can apply the gradient descent algorithm: 

1.  First, we randomly pick an initial p-vector of weights (e.g. following a normal distribution). 
2.  Then, we compute the gradient of the loss function in the initial p-vector. 
3.  The gradient direction indicates the direction to take in order to maximise the loss function. So, we take a small step in the opposite direction of gradient and we update weights' values accordingly using this update rule: 
![Octocat (https://raw.githubusercontent.com/ZiedHY/ZiedHY.github.io/ZiedHY-patch-1/UpdateRule.PNG)

4.  We move continuously until convergence to reach the lowest point of this landscape (local minima). 


This second article will be about **Sequence modeling with Neural Networks**. We will learn how to model sequences with a focus on Recurrent Neural Networks (RNNs) and their short-term memory and Long Short Term Memory (LSTM) and their ability to keep track of information throughout many timesteps.  

Let us go! 

# Sequence Modeling with Neural Networks 

## Context 

In the previous course, we saw how to use Neural Networks to model a dataset of many examples. The good news is that the basic architecture of Neural Networks is quite generic, whatever the application: a stacking of several perceptrons to compose complex hierarchical models and optimization of these models using gradient descent and backpropagation. 

Inspite of this, you have probably heard about Multilayer Perceptrons (MLPs), Convolutional Neural Networks (CNNs), Recurrent Neural Networks (RNNs), LSTM, Auto-Encoders, etc. These deep learning algorithms are different from each other. Each model is known to be particulary performant in some specific tasks, even though, fundamentally, they all share the same basic architecture. 

What makes the difference between them is their ability to be more suited for some data structures: dealing with text could be different from dealing with images, which in turn could be different from dealing with signals. 

In the balance of this article, we will focus on modeling **sequences** as a well-known data structure. 

Today's challenges in terms of quality of service and customer engagement revealed many applications of sequence modeling in day-to-day business practice:

*   Speech Recognition to listen to the voice of customers.
*   Machine Language Translation from diverse source languages to more common languages. 
*   Name entity/Subject extraction to find the main subject of the customer’s query once translated. 
*   Speech Generation to have conversational ability and engage with customers just like a human. 
*   Text Summarization of customer feedback to work on key challenges/pain points.   

In the car industry, self-parking is also a sequence modeling task because parking is a sequence of mouvements and the next movement depend on the other previous mouvements. 


**Add a reference to the application that will follow in this article**

## Introduction to Sequence Modeling  

Sequences are a data structure where each example could be seen as a series of data points. This sentence: "I am currently reading an article about sequence modeling with Neural Networks" is an example that consists of multiple words and words depend on each other. The same applies to medical records. One single medical record consists in many measurments across the time. It is the same for speech waveforms. 

***So why we need a different learning framework to model sequences and what are the special features that we are looking for in this framework?***

For illustration purposes and with no loss of generality, let us focus on text as a sequence of words to motivate this need for a different learning framework.  

In fact, machine learning algorithms typically require the text input to be represented as a ***fixed-length*** vector. Many operations needed to train the model (network) can be expressed through algebraic operations on the matrix of input feature values and the matrix of weights (think about a n-by-p design matrix, where n is the number of samples observed, and p is the number of variables measured in all samples). 

Perhaps the most common fixed-length vector representation for texts is the bag-of-words or bag-of-n-grams due to its simplicity, efficiency and often surprising accuracy. However, the bag-of-words (BOW) has many disadvantages:  

First, the word order is lost, and thus different sentences can have exactly the same representation, as long as the same words are used. Example: “The food was good, not bad at all.” vs “The food was bad, not good at all.”. 

Even though bag-of-n-grams considers the word order in short context, it suffers from data sparsity and high dimensionality. 

In addition, Bag-of-words and bag-of-n-grams have very little sense about the semantics of the words or more formally the distances between the words. This means that words “powerful”, “strong” and “Paris” are equally distant despite the fact that semantically, “powerful” should be closer to “strong” than “Paris”. 

Humans don’t start their thinking from scratch every second. As you read this article, you understand each word based on your understanding of previous words. Traditional neural networks can’t do this, and it seems like a major shortcoming. Bag-of-words and bag-of-n-grams as text representations does not allow to keep track of long-term dependencies inside the same sentence or paragraph. 

Another disadvantage of modeling sequences with traditional Neural Networks (e.g. Feedforward Neural Networks) is the fact of not sharing parameters across time. 

Let us take for example these two sentences : "On Monday, it was snowing" and "It was snowing on Monday". These sentences mean the same thing, though the details are in different parts of the sequence.  Actually, when we feed these two sentences 
into a Feedforward Neural Network for a prediction task, the model will assign different weights to "On Monday" at each moment in time. ***Things we learn about the sequence won’t transfer if they appear at different points in the sequence.***
Sharing parameters gives the network the ability to look for a given feature everywhere in the sequence, rather than in just a certain area. 

Thus, to model sequences, we need a specific learning framework able to: 
*   deal with variable-length sequences 
*   maintain sequence order 
*   keep track of long-term dependencies rather than cutting input data too short
*   share parameters across the sequence (so not re-learn things across the sequence) 

Recurrent neural networks (RNNs) could address this issue. They are networks with loops in them, allowing information to persist.

So, let us find out more about RNNs! 

## Recurrent Neural Networks

### How a Recurrent Neural Network works? 

A Recurrent Neural Network is architected in the same way as a "normal" Neural Network. We have some inputs, we have some hidden layers and we have some outputs. 

![Branching](C:\Users\zhajyahi\Documents\DeepLearning\MIT 6.S191\ZiedHY.github.io\Basic_Architecture_RNN.PNG)

The only difference is that each hidden unit is doing a slightly different function. So, let us take a look at this one hidden unit to see exactly what it is doing. 

A recurrent hidden unit computes a function of an input and its own previous output, also known as the cell state. For textual data, an input could be a word _x(i)_ in a sentence of _n_ words.  

![Branching](C:\Users\zhajyahi\Documents\DeepLearning\MIT 6.S191\ZiedHY.github.io\Hidden_Unit_RNN.PNG)

_W_ and _U_ are weight matrices and _tanh_ is the hyperbolic tangent function. 

Similarly, at the next step, it computes a function of the new input and its previous cell state: ***_s2_ = _tanh_(_Wx1_ + _Us1_)***. This function is similar to the function associated to hidden unit in a feed-forward Network. The difference,  proper to sequences, is that we are adding an additional term to incorporate its own previous state. 

A common way of viewing recurrent neural networks is by unfolding them across time. We can notice that ***we are using the same weight matrices _W_ and _U_ throughout the sequence. This solves our problem of parameter sharing***. We don't have new parameters for every point of the sequence. Thus, once we learn something, it can apply at any point in the sequence. 

![Branching](C:\Users\zhajyahi\Documents\DeepLearning\MIT 6.S191\ZiedHY.github.io\Unfolding_RNN.PNG)
 
The fact of not having new parameters for every point of the sequence also helps us ***deal with variable-length sequences***. 
In case of a sequence that has a length of 4, we could unroll this RNN to four timesteps. In other cases, we can unroll it to ten timesteps since the length of the sequence is not prespecified in the algorithm. By unrolling we simply mean that we write out the network for the complete sequence. For example, if the sequence we care about is a sentence of 5 words, the network would be unrolled into a 5-layer neural network, one layer for each word.

#### NB: 

*    _Sn_, the cell state at time _n_, can contain information from all of the past timesteps: each cell state is a function of the previous self state which in turn is a function of the previous cell state. ***This solves our issue of long-term dependencies***.  
*   The above diagram has outputs at each time step, but depending on the task this may not be necessary. For example, when predicting the sentiment of a sentence we may only care about the final output, not the sentiment after each word. Similarly, we may not need inputs at each time step. The main feature of an RNN is its hidden state, which captures some information about a sequence.


Now we saw how a single hidden unit works. But in a full network, we would have many hidden units and even many layers of many hidden units. So let us find out how do we train an RNN. 
 
### How do we train a Recurrent Neural Network? 

Let us say, for a set of speeches in English, we need the model to automatically convert the spoken language into text i.e. at each timestep, the model produces a prediction of a transcript (an output) on the basis of the part of speech at this timestep (the new input) and the previous transcript (the previous cell state). 

Naturally, because we have an output at every timestep, we can have a loss at every timestep. This loss reflects how much the predicted transcripts are close to the "official" transcripts.

![Branching](C:\Users\zhajyahi\Documents\DeepLearning\MIT 6.S191\ZiedHY.github.io\Loss_TimeStep_RNN.PNG)

The total loss is just the sum of the losses at every timestep.  

![Branching](C:\Users\zhajyahi\Documents\DeepLearning\MIT 6.S191\ZiedHY.github.io\Total_Loss_RNN.PNG)

Since the loss is a function of the network weights, our task it to find the set of weights _theta_ that achieve the lowest loss. For that, as explained in the first article "Introduction to Deep Learning", we we can apply ***the gradient descent algorithm with backpropagation (chain rule) at every timestep***, thus taking into account the additional time dimension. 

_W_ and _U_ are our two weight matrices. Let us try it out for _W_. 

Knowing that the total loss is the sum of the losses at every timestep, the total gradient is just the sum of the gradients at every timestep: 
 
![Branching](C:\Users\zhajyahi\Documents\DeepLearning\MIT 6.S191\ZiedHY.github.io\Derivative_Loss_RNN.PNG)

And now, we can focus on a single timestep to calculate the derivative of the loss with respect to _W_. 

![Branching](C:\Users\zhajyahi\Documents\DeepLearning\MIT 6.S191\ZiedHY.github.io\Graph_Chain_Rule_RNN.PNG)

Easy to handle: we just use backpropagation.

![Branching](C:\Users\zhajyahi\Documents\DeepLearning\MIT 6.S191\ZiedHY.github.io\Chain_Rule_RNN.PNG)

We remember that ***_s2_ = _tanh_(_Wx1_ + _Us1_)*** so s2 also depends on s1 and s1 also depends on W. ***This actually means that we can not just leave the the derivative of _s2_ with respect to _W_ as a constant. We have to expand it out farther.*** 

So how does _s2_ depend on _W_? 

It depends directly on W because it feeds right in (c.f. above formula of _s2_). We also saw that _s2_ depends on _s1_ which depends on W. And we can also see that _s2_ depends on _s0_ which also depends on W. 

![Branching](C:\Users\zhajyahi\Documents\DeepLearning\MIT 6.S191\ZiedHY.github.io\Derivative_Current_State_RNN.PNG)

Thus, the derivative of the loss with respect to _W_ could be written as follows: 

![Branching](C:\Users\zhajyahi\Documents\DeepLearning\MIT 6.S191\ZiedHY.github.io\Derivative_Loss_Chain_Rule_RNN.PNG)

We can see that the last two terms are basically summing the contributions of _W_ in previous timesteps to the error at timestep _t_. This is key to understand how we model long-term dependencies.  From one iteration to another, the gradient descent algorithm allows to shift network parameters such that they include contributions to the error from past timesteps.

For any timestep _t_, the derivative of the loss with respect to _W_ could be written as follows:

![Branching](C:\Users\zhajyahi\Documents\DeepLearning\MIT 6.S191\ZiedHY.github.io\Derivative_Loss_Chain_Rule_RNN_Generic.PNG)

So to train the model i.e. to estimate the weights of the network, we apply this same process of backpropagation through time for every weight (parameter) and then we use it in the process of gradient descent. 

### Why are Recurrent Neural Networks hard to train?

In practice RNNs are a bit difficult to train. To understand why, let’s take a closer look at the gradient we calculated above:

![Branching](C:\Users\zhajyahi\Documents\DeepLearning\MIT 6.S191\ZiedHY.github.io\Expand_Gradient_RNN.PNG)

We can see that as the gap between timesteps gets bigger, the product of the gradients gets longer and longer. But, what are each of these terms?  

![Branching](C:\Users\zhajyahi\Documents\DeepLearning\MIT 6.S191\ZiedHY.github.io\Vanishing_Gradient_RNN.PNG)

Each term is basically a product of two terms: transposed _W_ and a second one that depends on f'. 

*    Initial weights _W_ are usually sampled from standard normal distribution and then mostly < 1. 

*   It turns out (I won’t prove it here but [this paper](http://proceedings.mlr.press/v28/pascanu13.pdf) goes into detail) that
the second term is a Jacobian matrix because we are taking the derivative of a vector function with respect to a vector and 
its 2-norm, which you can think of it as an absolute value, ***has an upper bound of 1***. This makes intuitive sense because our tanh (or sigmoid) activation function maps all values into a range between -1 and 1, and the derivative f' is bounded by 1 (1/4 in the case of sigmoid). 

Thus, with small values in the matrix and multiple matrix multiplications, the ***gradient values are shrinking exponentially fast, eventually vanishing completely after a few time steps***. Gradient contributions from “far away” steps become zero, and the state at those steps doesn’t contribute to what you are learning: You end up not learning long-range dependencies. Vanishing gradients aren’t exclusive to RNNs. They also happen in deep Feedforward Neural Networks. It’s just that RNNs tend to be very deep (as deep as the sentence length in our case), which makes the problem a lot more common.

Fortunately, there are a few ways to combat the vanishing gradient problem. ***Proper initialization of the _W_ matrix*** can reduce the effect of vanishing gradients. So can regularization. A more preferred solution is to use ***ReLU*** instead of tanh or sigmoid activation functions. The ReLU derivative is a constant of either 0 or 1, so it isn’t as likely to suffer from vanishing gradients. 

An even more popular solution is to use Long Short-Term Memory (LSTM) or Gated Recurrent Unit (GRU) architectures. LSTMs were first proposed in 1997 and are the perhaps most widely used models in NLP today. GRUs, first proposed in 2014, are simplified versions of LSTMs. Both of these RNN architectures were explicitly designed to deal with vanishing gradients and efficiently learn long-range dependencies. We’ll cover them in the next section.




-----------------------------------------------------------------------------------------------------------

I have started reading about Deep Learning for over a year now through several articles and research papers that I came across mainly in LinkedIn, Medium and [Arxiv](https://arxiv.org/list/stat.ML/recent). 

When I virtually attended the MIT 6.S191 Deep Learning courses during the last few weeks [(Here is a link to the course site)](http://introtodeeplearning.com/), I decided to begin to put some structure in my understanding of Neural Networks through this series of articles. 

I will go through the first four courses: 
1.  Introduction to Deep Learning 
2.  Sequence Modeling with Neural Networks 
3.  Deep learning for computer vision - Convolutional Neural Networks 
4.  Deep generative modeling 

For each course, I will outline the main concepts and add more details and interpretations from my previous readings and my background in statistics and machine learning.  

Starting from the second course, I will also add an application on an open-source dataset for each course. 

That said, let's go! 

# Introduction to Deep Learning 

## Context 

Traditional machine learning models have always been very powerful to handle structured data and have been widely used by businesses for credit scoring, churn prediction, consumer targeting, and so on. 

The success of these models highly depends on the performance of the feature engineering phase: the more we work close to the business to extract relevant knowledge from the structured data, the more powerful the model will be. 

When it comes to unstructured data (images, text, voice, videos), hand engineered features are time consuming, brittle and not scalable in practice. That is why Neural Networks become more and more popular thanks to their ability to automatically discover the representations needed for feature detection or classification from raw data. This replaces manual feature engineering and allows a machine to both learn the features and use them to perform a specific task.

Improvements in Hardware (GPUs) and Software (advanced models / research related to AI) also contributed to **deepen the learning** from data using Neural Networks.  

## Basic architecture  

The fundamental building block of Deep Learning is the **Perceptron** which is a single neuron in a Neural Network. 

Given a finite set of _m_ inputs (e.g. _m_ words or _m_ pixels), we multiply each input by a weight (_theta 1_ to _theta m_) then we sum up the weighted combination of inputs, add a bias and finally pass them through a non-linear activation function. That produces the output _Yhat_. 

![Branching](https://raw.githubusercontent.com/ZiedHY/ZiedHY.github.io/ZiedHY-patch-1/Perceptron.PNG)

*   The bias _theta 0_ allows to add another dimension to the input space. Thus, the activation function still provide an output in case of an input vector of all zeros. It is somehow the part of the output that is independent of the input.
*   The purpose of activation functions is to introduce non-linearities into the network. In fact, linear activation functions produce linear decisions no matter the input distribution. Non-linearities allow us to better approximate arbitrarily complex functions. Here some examples of common activation functions: 
![Octocat](https://raw.githubusercontent.com/ZiedHY/ZiedHY.github.io/ZiedHY-patch-1/ActivationFunctions.PNG)

Deep Neural Networks are no more than a **stacking** of multiple perceptrons (hidden layers) to produce an output. 
![Branching](https://raw.githubusercontent.com/ZiedHY/ZiedHY.github.io/ZiedHY-patch-1/FullyConnected.PNG)

Now, once we have understood the basic architecture of a deep neural network, let us find out how it can be used for a given task. 

## Training a Neural Network 

Let us say, for a set of X-ray images, we need the model to automatically distinguish those that are related to a sick patient from the others. 

For that, machine learning models, like humans, need to learn to differentiate between the two categories of images by **observing** some images of both sick and healthy individuals. Accordingly, they automatically understand patterns that better describe each category. This is what we call **the training phase**.  

Concretely, a pattern is a weighted combination of some inputs (images, parts of images or other patterns). Hence, **the training phase is nothing more than the phase during which we estimate the weights (also called parameters) of the model.** 

When we talk about estimation, we talk about an **objective function** we have to optimize. This function shall be constructed to best reflect the performance of the training phase. When it comes to prediction tasks, this objective function is usually called **loss function** and measures the cost incurred from incorrect predictions. When the model predicts something that is very close to the true output then the loss function is very low, and vice-versa. 

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
![Octocat](https://raw.githubusercontent.com/ZiedHY/ZiedHY.github.io/ZiedHY-patch-1/UpdateRule.PNG)

4.  We move continuously until convergence to reach the lowest point of this landscape (local minima). 

#### NB: 

*   In the update rule, _Etha_ is the **learning rate** and determines how large is the step we take in the direction of our gradient. Its choice is very important since modern neural network architectures are extremly non-convex. If the learning rate is too low, the model could stuck in a local minimum. If it is too large it could diverge. **Adaptive learning rates** could be used to adapt the learning rate value for each iteration of the gradient. For more detailed explanation please read this [overview of gradient descent optimization algorithms by Sebastian Ruder](https://arxiv.org/pdf/1609.04747.pdf).

*   To compute the gradient of the loss function in respect of a given vector of weights, we use **backpropagation**. 
Let us consider the simple neural network above. It contains one hidden layer and one output layer. We want to compute the gradient of the loss function with respect to each parameter, let us say to _theta 1_. For that, we start by applying the chain rule because J(_theta_) is only dependent on _Yhat_. And then, we apply the chain rule one more time to backpropagate the gradient one layer further. We can do this, for the same reason, because _z1_ (hidden state) is only depend on the input _x_ and _theta 1_. 
Thus, the backpropagation consists in **repeating this process for every weight in the network using gradients from later layers**. 

![Octocat](https://raw.githubusercontent.com/ZiedHY/ZiedHY.github.io/ZiedHY-patch-1/Backpropagation.PNG)

## Neural Networks in practice:

*  In the presence of a large dataset, the computation of the gradient in respect to each weight can be very expensive (think about the chain rule in backpropagation). For that, we could compute the gradient on a subset of data (mini-batch) and use it as an estimate of the true gradient. This gives a more accurate estimation of the gradient than the stochastic gradient descent (SGD) which randomly takes only one observation and much more faster than calculating the gradient using all data. Using mini-batches for each iteration leads to fast training especially when we use different threads (GPU's). We can parallelize computing for each iteration: a batch for each weight and gradient is calculated in a seperate thread. Then, calculations are gathered together to complete the iteration.  

*  Juste like any other "classical" machine learning algorithm, Neural Networks could face the problem of overfitting. Ideally, in machine learning, **we want to build models that can learn representations from a training data and still generalize well on unseen test data**. Regularization is a technique that constrains our optimization problem to discourage complex models (i.e. to avoid memorizing data). When we talk about regularization, we generally talk about **Dropout** which is the process of randomly dropping out some proportion of the hidden neurals in each iteration during the training phase (dropout i.e. set associated activations to 0) and/or **Early stopping** which consists in stopping training before we have a chance to overfit. For that, we calculate the loss in training and test phase relative to the number of training iterations. We stop learning when the loss function in the test phase starts to increase. 

## Conclusion:

This first article is an introduction to Deep Learning and could be summarized in 3 key points: 

1.  First, we have learned about the fundamental building block of Deep Learning which is the Perceptron.  
2.  Then, we have learned about stacking these perceptrons together to compose more complex hierarchical models and we learned how to mathematically optimize these models using backpropagation and gradient descent. 
3.  Finally, we have seen some practical challenges of training these models in real life and some best practices like adaptive learning, batching and regularization to combat overfitting. 

The next article will be about **Sequence modeling with Neural Networks**. We will learn how to model sequences with a focus on Recurrent Neural Networks (RNNs) and their short-term memory and Long Short Term Memory (LSTM) and their ability to keep track of information throughout many timesteps.  

Stay tuned! 




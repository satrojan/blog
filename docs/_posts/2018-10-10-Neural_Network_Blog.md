---
layout: post
title: Conceptual Neural Network
---
Neural networks are currently the frontier for data science. Exploring and understanding neural nets are going to be critical for the future. Even today Neural Networks power devices in our homer such as Amazon's Alexa. These uses will continue to expand and propagate through our lives and future.

In my attempt to learn more about Neural Networks I came across the following [article](http://neuralnetworksanddeeplearning.com/chap4.html) By Michael Nielsen that did a very good job of describing the basics of neural networks in a mathematical way. I decided to take a more programmatical approach and attempt to code the solution in python.

To begin we are going to be modeling a single input neural network with five hidden nodes following the function $$f(x)=0.2+0.4x^2+0.3x\sin(15x)+0.05\cos(50x)$$ That looks scary but it's not that bad. Let's run it in code and see what it looks like.


```python
import math, numpy as np
import seaborn as sns

sns.set(style="darkgrid")
```


```python
x = np.arange(0,1,.01) #inputs
y = .2+(0.4*x**2)+(0.3*x*np.sin(15*x))+(0.05*np.cos(50*x))
sns.set(style="darkgrid")
sns.lineplot(x=x, y=y);
```


![png](/blog/docs/assets/images/p_nnc/output_2_0.png)


Trying to plot this out exactly would be difficult. Fortunately we are only interested in an approximation of the function. Now the internals of an actual neuron in a neural network are more complex involving different weights and biases that differ when we train our neural network. The article above does a wonderful job describing this as well at allowing you to train a neural network by hand.

I will now emulate this neural network in python.


```python
y_m = []
for i in x:
    if 0<=i<.2:
        y_m.append(-1.3)
    elif 0.2<=i<.4:
        y_m.append(-1.5)
    elif 0.4<=i<.6:
        y_m.append(-.5)
    elif 0.6<=i<.8:
        y_m.append(-1)
    elif 0.8<=i<=1:
        y_m.append(.9)
```


```python
sns.set(style="darkgrid")
sns.lineplot(x=x, y = y_m); # neural network before applying inveres sigmoid.
```


![png](/blog/docs/assets/images/p_nnc/output_5_0.png)


This is the initial trained neural network output. Once we get this output we will apply the reverse sigmoid function to get our true output.


```python
def sigmoid_activation(x):
    return 1/(1+np.e**-x)

y_output = [sigmoid_activation(x) for x in y_m]

sns.set(style="darkgrid")
sns.lineplot(x=x, y = y)
sns.lineplot(x=x, y = y_output);
```


![png](/blog/docs/assets/images/p_nnc/output_7_0.png)


Our output is far from perfect but with only five nodes we cannot be asking for too much from our emulated network. Let's quickly calculate the MSE.


```python
.001*np.sum(np.square(np.subtract(y_output,y))) #MSE
```




    0.0011450802059996784



We can see that our MSE is very low showing our emulated neural network is a good fit for our initial equation. We can greatly improve our results in the emulated neural network simply by adding more nodes. An actual neural network can increase our results by adding a second hidden layer and attempting to manipulate other hyperparameters but that is something that we cannot emulate using simple if statements.

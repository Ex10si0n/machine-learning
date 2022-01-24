# Implementation of Neural Networks from scratch

In the past 10 years, the best-performing artificial-intelligence systems — such as the speech recognizers on smartphones or Google’s latest automatic translator — have resulted from a technique called “Deep Learning.”

Deep learning is in fact a new name for an approach to artificial intelligence called neural networks, which have been going in and out of fashion for more than 70 years. **Neural networks** were first proposed in 1944 by Warren McCullough and Walter Pitts.

**Neural network** is an interconnected group of nodes, inspired by a simplification of neurons in a brain. It works similarly to the human brain’s neural network. A “neuron” in a neural network is a mathematical function that collects and classifies information according to a specific architecture. The network bears a strong resemblance to statistical methods such as curve fitting and regression analysis.&#x20;

In this course, we’ll understand how neural networks work while implementing one from scratch in Python.

### Basic Component: Neurons

First, we have to talk about neurons, the basic unit of a neural network. **A neuron takes inputs, does some math with them, and produces one output**. Here’s what a 2-input neuron looks like:

![Neuron](https://victorzhou.com/a74a19dc0599aae11df7493c718abaf9/perceptron.svg)

3 things are happening here. First, each input is multiplied by a weight:&#x20;

$$
x1→x1∗w1x1​→x1​∗w1​x2→x2∗w2x2​→x2​∗w2​
$$

Next, all the weighted inputs are added together with a bias bb:&#x20;

$$
(x1∗w1)+(x2∗w2)+b(x1​∗w1​)+(x2​∗w2​)+b
$$

Finally, the sum is passed through an activation function:&#x20;

$$
y=f(x1∗w1+x2∗w2+b)y=f(x1​∗w1​+x2​∗w2​+b)
$$

\
\

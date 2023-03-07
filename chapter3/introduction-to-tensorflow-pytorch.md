# Introduction to Tensorflow

Copyright 2019 The TensorFlow Authors

```
#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

## TensorFlow 2 quickstart for beginners <a href="#tensorflow-2-quickstart-for-beginners" id="tensorflow-2-quickstart-for-beginners"></a>

| [![](https://camo.githubusercontent.com/59d636752813948be5871700a7cfe31cf657164cf3fb9294b3e33eaf45488afb/68747470733a2f2f7777772e74656e736f72666c6f772e6f72672f696d616765732f74665f6c6f676f5f333270782e706e67)View on TensorFlow.org](https://www.tensorflow.org/tutorials/quickstart/beginner) | [![](https://camo.githubusercontent.com/756e8e5187b778c7b7440cce63e1ca5069313fea0abddc151a92f5b5f536f471/68747470733a2f2f7777772e74656e736f72666c6f772e6f72672f696d616765732f636f6c61625f6c6f676f5f333270782e706e67)Run in Google Colab](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/quickstart/beginner.ipynb) | [![](https://camo.githubusercontent.com/a7636a071984705e0b7a669e2bcb64246292e25a2ac9174fdd7c5eaae363c197/68747470733a2f2f7777772e74656e736f72666c6f772e6f72672f696d616765732f4769744875622d4d61726b2d333270782e706e67)View source on GitHub](https://github.com/tensorflow/docs/blob/master/site/en/tutorials/quickstart/beginner.ipynb) | [![](https://camo.githubusercontent.com/3cf80682de19783a0ab31047da32a08b5d62312c3ccd0aa055e7b0576a98a830/68747470733a2f2f7777772e74656e736f72666c6f772e6f72672f696d616765732f646f776e6c6f61645f6c6f676f5f333270782e706e67)Download notebook](https://storage.googleapis.com/tensorflow\_docs/docs/site/en/tutorials/quickstart/beginner.ipynb) |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |

This short introduction uses [Keras](https://www.tensorflow.org/guide/keras/overview) to:

1. Load a prebuilt dataset.
2. Build a neural network machine learning model that classifies images.
3. Train this neural network.
4. Evaluate the accuracy of the model.

This tutorial is a [Google Colaboratory](https://colab.research.google.com/notebooks/welcome.ipynb) notebook. Python programs are run directly in the browser—a great way to learn and use TensorFlow. To follow this tutorial, run the notebook in Google Colab by clicking the button at the top of this page.

1. In Colab, connect to a Python runtime: At the top-right of the menu bar, select _CONNECT_.
2. Run all the notebook code cells: Select _Runtime_ > _Run all_.

### Set up TensorFlow <a href="#set-up-tensorflow" id="set-up-tensorflow"></a>

```
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
```

If you are following along in your own development environment, rather than [Colab](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/quickstart/beginner.ipynb), see the [install guide](https://www.tensorflow.org/install) for setting up TensorFlow for development.

Note: Make sure you have upgraded to the latest `pip` to install the TensorFlow 2 package if you are using your own development environment. See the [install guide](https://www.tensorflow.org/install)for details.

### Load a dataset <a href="#load-a-dataset" id="load-a-dataset"></a>

```
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
```

### Build a machine learning model <a href="#build-a-machine-learning-model" id="build-a-machine-learning-model"></a>

```
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])
```

For each example, the model returns a vector of [logits](https://developers.google.com/machine-learning/glossary#logits) or [log-odds](https://developers.google.com/machine-learning/glossary#log-odds) scores, one for each class.

```
predictions = model(x_train[:1]).numpy()
predictions
```

The `tf.nn.softmax` function converts these logits to _probabilities_ for each class:

```
tf.nn.softmax(predictions).numpy()
```

Note: It is possible to bake the `tf.nn.softmax` function into the activation function for the last layer of the network. While this can make the model output more directly interpretable, this approach is discouraged as it's impossible to provide an exact and numerically stable loss calculation for all models when using a softmax output.

Define a loss function for training using `losses.SparseCategoricalCrossentropy`, which takes a vector of logits and a `True` index and returns a scalar loss for each example.

```
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
```

This loss is equal to the negative log probability of the true class: The loss is zero if the model is sure of the correct class.

This untrained model gives probabilities close to random (1/10 for each class), so the initial loss should be close to `-tf.math.log(1/10) ~= 2.3`.

```
loss_fn(y_train[:1], predictions).numpy()
```

Before you start training, configure and compile the model using Keras `Model.compile`. Set the [`optimizer`](https://www.tensorflow.org/api\_docs/python/tf/keras/optimizers) class to `adam`, set the `loss` to the `loss_fn`function you defined earlier, and specify a metric to be evaluated for the model by setting the `metrics` parameter to `accuracy`.

```
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
```

### Train and evaluate your model <a href="#train-and-evaluate-your-model" id="train-and-evaluate-your-model"></a>

```
model.fit(x_train, y_train, epochs=5)
```

The `Model.evaluate` method checks the models performance, usually on a "[Validation-set](https://developers.google.com/machine-learning/glossary#validation-set)" or "[Test-set](https://developers.google.com/machine-learning/glossary#test-set)".

```
model.evaluate(x_test,  y_test, verbose=2)
```

The image classifier is now trained to \~98% accuracy on this dataset. To learn more, read the [TensorFlow tutorials](https://www.tensorflow.org/tutorials/).

If you want your model to return a probability, you can wrap the trained model, and attach the softmax to it:

```
probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])
```

```
probability_model(x_test[:5])
```

### Conclusion <a href="#conclusion" id="conclusion"></a>

Congratulations! You have trained a machine learning model using a prebuilt dataset using the [Keras](https://www.tensorflow.org/guide/keras/overview) API.

For more examples of using Keras, check out the [tutorials](https://www.tensorflow.org/tutorials/keras/). To learn more about building models with Keras, read the [guides](https://www.tensorflow.org/guide/keras). If you want learn more about loading and preparing data, see the tutorials on [image data loading](https://www.tensorflow.org/tutorials/load\_data/images) or [CSV data loading](https://www.tensorflow.org/tutorials/load\_data/csv).


---
layout: post
title: Regularization in Artificial Neural Networks
subtitle:
gh-repo: skhabiri/ML-ANN/tree/main/module4-Deploy
gh-badge: [star, fork, follow]
tags: [Machine Learning, Neural Network, Regularization, TensorFlow, Keras]
image: /assets/img/post10/post10_regularization.png
comments: false
---

Neural Networks are highly parameterized models and can be easily overfit to the training data. The most salient way to combat this problem is with regularization techniques. A common technique to prevent overfitting is to use `EarlyStopping`. This strategy will prevent your weights from being updated well past the point of their peak usefulness. We can also combine `EarlyStopping`, `Weight Decay` and `Dropout`, or use `Weight Constraint` instead of `Weight Decay`, which accomplishes similar ends.

### Dataset
Our dataset is fashion_mnist with 60k train and 10k for test. There are 10 classes with equal distribution.
```
from tensorflow.keras.datasets import fashion_mnist

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X_train.shape, X_test.shape
```
((60000, 28, 28), (10000, 28, 28))
Here is a sample of images of different classes:

<img src="../assets/img/post9/post10_samples.png" />

Next, we normalize and flatten the input tensor to make the array one dimensional. To convert a multidimensional tensor into a 1-D tensor, we can use `Flatten` layer from `tensorflow.keras.layers`.
```
X_train, X_test = X_train / 255., X_test / 255.
```


```
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.layers import ReLU
import tensorflow as tf
import os

# create subdirectory
logdir = os.path.join("logs", "EarlyStopping-Loss")

# frequency (in epochs) at which to compute activation and  
# weight histograms for the layers of the model.
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

# patience refers to # of epochs with no improvement
stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.01, patience=3)


model = tf.keras.Sequential([
    # Flattens multi two dimensional input array of (28,28) onto a 1D output array of (784,)
    # Flatten layer also acts as input layer. hence input_shape parameter is defined
    Flatten(input_shape=(28,28)),  # Treats as 784x1
    
    # Since Dense is not the input layer here, we don't need to specify input_shape or input_dim
    Dense(128),
    # parametric ReLu activation function shows up as a layer rather than a parameter for Dense
    ReLU(negative_slope=.01),
    Dense(128),
    ReLU(negative_slope=.01),
    Dense(128),
    ReLU(negative_slope=.01),
    
    # output layer, softmax converts the class values to probability
    Dense(10, activation='softmax')
])

# scce expect the target classes to be integer numbers not OHE
model.compile(loss='sparse_categorical_crossentropy', optimizer='nadam',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=99, 
          validation_data=(X_test,y_test),
          callbacks=[tensorboard_callback, stop])
```










### Conclusion
We presented three different approaches for hyperparamter tuning of a neural network. Using Keras sklearn wrapper, HParams Dashboard in TensorBoard, and keras-tuner. all three methods were applied on MNIST dataset to classify the digit labels. The tuned model achieved test accuracy of about 0.97.

### links
- [Github repo](https://github.com/skhabiri/ML-ANN/tree/main/module3-Tune)
- [Keras](https://keras.io)
- [TensorFlow](https://www.tensorflow.org)

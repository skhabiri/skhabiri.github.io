---
layout: post
title: Sketch Classification With Neural Networks
subtitle: Classification of QuickDraw Dataset
gh-repo: skhabiri/ML-ANN/tree/main/module2-Train
gh-badge: [star, fork, follow]
tags: [Machine Learning, Neural Network, QuickDraw, Classification, TensorFlow, Keras]
image: /assets/img/post7/post7_reddit.jpg
comments: false
---

We are going to use TensorFlow Keras and a sample of the [Quickdraw dataset](https://github.com/googlecreativelab/quickdraw-dataset) to build a sketch classification model. The dataset has been sampled to only 10 classes and 10000 observations per class. We will build a baseline classification model then run a few experiments with different optimizers and learning rates.

### Load dataset
Our data is in Numpy's compressed array (npz) format. We need to load it from a url address. First, we need to import the following modules.
```
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import wget
```
We load, shuffle and split the data into train and test with a ratio of 0.2:
```
def load_quickdraw10(path):
  wget.download(path)
  data = np.load('quickdraw10.npz')
  X = data['arr_0']
  y = data['arr_1']

  print(X.shape)
  print(y.shape)

  X, y = shuffle(X, y)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  
  return X_train, y_train, X_test, y_test

path = 'https://github.com/skhabiri/ML-ANN/raw/main/data/quickdraw10.npz'
X_train, y_train, X_test, y_test = load_quickdraw10(path)
```
(100000, 784)
(100000,)
We have 10 classes with 10K samples for each class. Each input sample image is represented by an array of 784 dimensions. Array values are from 0 to 255. As a good practice we normalize the input array values.
```
xmax = X_train.max()
X_train = X_train / xmax
X_test = X_test / xmax
X_train.max()
```
1.0

* The selected classes are:
`class_names = ['apple', 'anvil', 'airplane', 'banana', 'The Eiffel Tower', 'The Mona Lisa', 'The Great Wall of China', 'alarm clock', 'ant', 'asparagus']`

### Build the model
We write a function to returns a compiled TensorFlow Keras Sequential Model suitable for classifying the QuickDraw-10 dataset. We leave `learning rate` and  `optimizer` as hyperparamters to tune later.
```
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import *

def create_model(optim, lr=0.01):
    """
    optim: class of the optimizer. values: [Adadelta, Adagrad, Adam, Ftrl, SGD]
    """
    opt = optim(learning_rate=lr)
    model = Sequential(
        [
        #  784 inputs + 1 bias connect to 32 1st layer Hiddent neurons
        Dense(32, activation='relu', input_dim=784),
        #  32 1st-H-Neurons + 1 bias connected to 32 2'nd layer H-Neurons
        Dense(32, activation='relu'),
        #  32 2nd-H-neurons connect to 10 Output neurons
        Dense(10, activation='softmax')       
        ]
    )
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
```
We have 784 inputs, two dense layers each with 32 neurons, and 10 output classes. Including the bias node at each layer, total number of weights are `784+1 * 32+1 * 32+1 * 10`. Since this is a multilable classification with integer classes, we use sparse_categorical_crossentropy.

Let's sweep different hyperparameters and review its effect on the model accuracy. Below is a function to sweep values of a hyperparameter and fit the model. This would allow us to examine the sensitivity of the model to a particular hyperparameter.
```
def fit_param(param_lst, key, **kwargs):
    """
    This function fits a ANN created by create_model() while sweeping a parameter
    param_list: list of values for the parameter
    key: string key for the parameter. Values: "lr", "batch_size", "epochs", "optimizer"
    return: a dictionary with 
    {f"{par}_": [fitted model, fitted result],
    "key": key, "param_lst": param_lst}
    model_dict[f"{par}_"][0] is the model
    model_dict[f"{par}_"][1] is the fit result
    kwargs: all the keyword arguments that have been used in the function  
    """

    # initialize **kwargs:
    if not kwargs:
        kwargs = {"lr": 0.1, "batch": 128, "epoch": 5, "optimizer": Adam}
    
    model_dict={}
    model_dict["key"] = key
    model_dict["param_lst"] = param_lst
    for par in param_lst:
        kwargs[key] = par
        print(f"********* Fitting for {key}={kwargs[key]} *********")
        print(f""" Fitting for lr, batch, epoch, optimizer=
        {kwargs["lr"]}, {kwargs["batch"]}, {kwargs["epoch"]}, {kwargs["optimizer"]}""")
        # Initialize the dictionary
        model_dict.setdefault(f"{par}_", [None, None])
        model_dict[f"{par}_"][0] = create_model(kwargs["optimizer"], kwargs["lr"])
        model_dict[f"{par}_"][1] = model_dict[f"{par}_"][0].fit(
            X_train, y_train,
            # Hyperparameters!
            epochs=kwargs["epoch"], 
            batch_size=kwargs["batch"], 
            validation_data=(X_test, y_test))
    
    return model_dict
```
This function returns the trained model as well as the fitted results for each value of the swept parameter. Let's now fit the model for different values of each hyperparameter:
```
batch_lst = [8, 32, 512, 4096]
lr_lst = [0.0001, 0.01, 0.5, 1]
opt_lst = [Adadelta, Adam, SGD]

params_dic = { 
    "optimizer": [opt_lst, None, None, {"lr":0.01, "batch":32, "epoch":25, "optimizer":SGD}],
    "batch_size": [batch_lst, None, None, {"lr":0.01, "batch":32, "epoch":25, "optimizer":SGD}],
              "lr": [lr_lst, None, None, {"lr":0.01, "batch":32, "epoch":25, "optimizer":SGD}],
             }
df_lst = []

for key, val in params_dic.items():
    
    kwargs = params_dic[key][3]

    # create model
    params_dic[key][1] = fit_param(params_dic[key][0], key, **kwargs)
```
Now we can plot the validation accuracy for each hyper paramter and review its effect on the trained model.

<p float="left">
  <img src="../assets/img/post8/post8_optimizer.png" width="350" />
  <img src="../assets/img/post8/post8_batchsize.png" width="350" /> 
  <img src="../assets/img/post8/post8_lr.png" width="350" />
</p>

Among different choices for `optimizer` engine, SGD and Adam seems to be more efficient for this dataset. The choices of `batch size` does not seem to be critical to the accuracy of the model. The entire input X is divided into batches of size n and the neural network is trained on each batch of n samples. In our perceptron network, weights W's, and biases b's get updated at the end of each batch. Once all batches in a training dataset are trained the epoch counter goes up and we create another set of batches randomly and exclusively (like Kfold) and re-train based on each of the new batches again.
Considering two extreme cases, in stochastic gradient descent, batch size is set to one sample. Hence the accuracy of each update is low. However, number of updates per epoch are maximum, as there is one back-propagation update per batch. That resuls in long computing time and noisy training trend since the updates are done based on individual samples. On the other side for batch size gradient descent (GD), we have one batch per epoch, or in other word, the size of the batch is equal to the entire training set. Hence the epoch looks at the same set of data repeatedly and makes an update on every epoch run. Here since back propagation takes place after looking at the entire training set, the updates are more generalized and less noisy. Due to less number of batches per epoch, one batch per epoch, runtime is shorter. However, we need a large memory to process the entire dataset in one batch, and with large dataset that is not feasible.
As for `learning rate`, a large number like 1 fails to converge, while a very small number such as 0.0001 underfits and needs more epochs to train. However, a learning rate between 0.01 to 0.5 yields reasonable results.













































Here is a sample of the data:

<table border="1" class="dataframe" style="overflow-x: scroll;display: block; max-height: 300px;"><thead><tr style="text-align: right;"><th></th><th>subreddit_name</th><th>subreddit_id</th><th>title</th><th>text</th></tr></thead><tbody><tr><th>0</th><td>literature</td><td>2qhps</td><td>James Franco's poems: hard to forgive</td><td></td></tr><tr><th>1</th><td>technology</td><td>2qh16</td><td>Predator Drone Spotted in Minneapolis During George Floyd Protests</td><td></td></tr><tr><th>2</th><td>DIY</td><td>2qh7d</td><td>I restored a $5 Kitchen Aid mixer I found at the thrift store</td><td></td></tr><tr><th>3</th><td>news</td><td>2qh3l</td><td>Alabama just passed a near-total abortion ban with no exceptions for rape or incest</td><td></td></tr><tr><th>4</th><td>Parenting</td><td>2qhn3</td><td>I thought my 6 year old was doing one of his math activities on the tablet, but nah</td><td>My 6 year old has a bunch of new apps and activities that his teacher sent us to put on his tablet. He's been occasionally asking me, from the other room, the answers to different math problems, like what's 12+7 or what's 22-8.  I'm like sweet he's doing his math! Nope. He's trying to bypass the parental locks on kids YouTube so he can watch shit that is blocked. He keeps exiting out and going back in which is I assume why he had to ask multiple times.</td></tr></tbody></table>

The database contains 51610 rows and 4 columns.
```
 #   Column          Non-Null Count  Dtype 
---  ------          --------------  ----- 
 0   subreddit_name  51610 non-null  object
 1   subreddit_id    51610 non-null  object
 2   title           51610 non-null  object
 3   text            51610 non-null  object
 ```
Let's check the unique categories in our collected data:
```
 subreddit_names = data['subreddit_name'].unique()
len(subreddit_names), subreddit_names
 ```
 Our created dataset consists of 53 categories.
 ```
 (53,
 array(['literature', 'technology', 'DIY', 'news', 'Parenting', 'cars',
        'WTF', 'MachineLearning', 'socialskills', 'Art', 'biology',
        'politics', 'personalfinance', 'sports', 'worldpolitics',
        'Documentaries', 'food', 'LifeProTips', 'movies',
        'TwoXChromosomes', 'nottheonion', 'mildlyinteresting', 'Health',
        'AskReddit', 'history', 'Cooking', 'Music', 'Fitness',
        'GetMotivated', 'Design', 'gaming', 'entertainment', 'television',
        'books', 'JusticeServed', 'math', 'investing', 'science',
        'camping', 'Coronavirus', 'PublicFreakout', 'travel', 'funny',
        'HomeImprovement', 'scifi', 'worldnews', 'AdviceAnimals',
        'programming', 'gadgets', 'conspiracy', 'space', 'Showerthoughts',
        'announcements'], dtype=object))
 ```
 Let's look at number of posts per subreddit category.
 ```
sns.histplot(
    x=data['subreddit_name'].astype('category').cat.codes, 
    bins=data['subreddit_id'].nunique(),
    kde=True)
 ```
 <img src= "../assets/img/post7/post7_subredditposts.png">

Ideally we would use the entire dataset for the training. However for practical reasons that would substantially increase the size of the serialized model and complicate the deployment of the model. For this reason we are going to use a smaller subset of the dataset for training. 
The above graph shows that we have about 1000 posts per subreddit category as expected. However some the posts might have small amount of text that would not be sufficient for our natural language processing. 
Hence we choose the posts that have enough text content. Later on we are going to choose only the categories (features) that  have enough number of posts (instances) to train on.

To get an idea of the posts lengths, let's plot the average length of posts per subreddit category.
```
post_mean = data1.groupby(by='subreddit_name').apply(lambda x: x['text_length'].mean())
plt.figure(figsize=(8,4))
ax = sns.barplot(x=post_mean.index, y=post_mean.values)
ax.set(xlabel='Subreddit Category', ylabel='Average length of posts')
ax.set(xticklabels=[])
plt.show()
```
<img src= "../assets/img/post7/post7_postlength_avg.png">

The above graph shows the average length of posts is not the same in different subreddit categories.
After filtering the low count subreddit categories we end up with 44 categories and 4400 posts. We take a note that in a production setup we need more training data to achieve a reliable results.



### Conclusion
In this work we built a machine learning model using NLP techniques and optimize that we scikit-learn RandomSearchCV() to predict a subreddit category for a given post. We used a python wrapper for Reddit API, PRAW, to create a database of subreddit posts from the categories of interest. Afte cleaning the data, we fit and tuned three different models and compared their performances. Other than the accuracy score we ran an article a sample input and used one of the models to get similar articles from the training set. We also used the sample input article to predict top subreddit categories that are related to the article. The serialized model can be deployed to a datascience API in order to build a full stack application.

### links
- [Github repo](https://github.com/skhabiri/SubReddit-Recommender)
- [PRAW](https://praw.readthedocs.io)
- [SciKit-Learn](https://scikit-learn.org/stable/getting_started.html)
- [spaCy](https://spacy.io/)

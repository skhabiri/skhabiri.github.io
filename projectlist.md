

Recent ML/DataScience Work

### Natural Language Processing (NLP)
  - *Tokenization and visualization of customer reviews:*
    - Performed frequency based and w2v tokenization on review texts of “Datafiniti Amazon consumer reviews” and “Yelp coffee shop reviews” datasets separately. The tokens were statistically trimmed and visualized by Squarify library. [github](https://github.com/skhabiri/ML-NLP/tree/main/module1-text-data)
  - *Text Vectorization and similarity search:* 
    - Vectorized a dataset of 401 articles from BBC website as a Bag of Words, and performed similarity search on the queried articles. [github](https://github.com/skhabiri/ML-NLP/blob/main/module2-vector-representations/Vector_Representations-412.ipynb) 
    - Performed similarity search on a dataset of job listing descriptions with html tags. Worked with Beautiful Soup, a Python library to pull data out of HTML files for data wrangling purposes. [github](https://github.com/skhabiri/ML-NLP/blob/main/module2-vector-representations/Vector_Representations-412a.ipynb)
  - *Document Classification:*
    - Performed binary classification on a newsgroups dataset with two categories of “alt.atheism” and “talk.religion.misc”, using Latent Semantic Indexing (LSI).  After hyper-parameter tuning an accuracy of 0.91 was achieved. [github](https://github.com/skhabiri/ML-NLP/tree/main/module3-document-classification)
    - Performed document classification on whiskey review dataset with three target label classes. [github](https://github.com/skhabiri/ML-NLP/blob/main/module3-document-classification/Document_Classification-413a.ipynb)
  - *Topic modeling and sentiment analysis:*
    - Used Gensim library to create a dictionary of words and tokenize reviews in an IMBD reviews dataset. Derived topic composition of each review and performed sentiment analysis for each primary topic. Visualized the topics using pyLDAvis library. [github](https://github.com/skhabiri/ML-NLP/tree/main/module4-topic-modeling)
    - Performed topic modeling on three categories of newsgroup dataset. Topic perplexity and topic coherence in applied LDA models were evaluated. [github](https://github.com/skhabiri/ML-NLP/blob/main/module4-topic-modeling/Topic_Modeling-414a.ipynb)
  - *A Subreddit Recommendation Engine Using NLP techniques:*
    - Built a recommendation engine that takes a text and recommends a subreddit category based on the content. Praw, a reddit api for python was utilized to pull data from reddit into a local SQL database. Achieved 0.53 accuracy for 44 subreddit categories. [github](https://github.com/skhabiri/SubReddit-Recommender), [blog post](https://skhabiri.com/2020-10-20-Building_A_Subreddit_Recommendation_Engine_Using_Machine_Learning_Techniques/)

### Feedforward Artificial Neural Networks (Perceptron)
  - *Supervised learning using artificial neural networks:*
    - Performed hyper-parameter tuning on MNIST dataset. Achieved validation accuracy of 0.95. [github](https://github.com/skhabiri/ML-ANN)
    - Trained a feed forward NN on Quickdraw dataset containing 100K drawing in ten classes. Applied common techniques to avoid overfitting. Achieved accuracy score of 0.82. [github](https://github.com/skhabiri/ML-ANN/blob/main/module3-Tune/ann_tune-423a.ipynb), [blog post](https://skhabiri.com/2021-01-12-Sketch-Classification-with-Neural-Networks/)

### Recurrent Neural Network (RNN)
  - *Sentiment Classification:*
    - Trained a RNN consists of Keras Embedding and LSTM layers to classify review sentiments for an IMBD Reviews dataset. Resulting in accuracy score of 0.83. [github](https://github.com/skhabiri/ML-DeepLearning/tree/main/module1-rnn-and-lstm)
  - *Text generation:*
    - Trained a RNN on a dataset containing news articles to generate text given a seed phrase. [github](https://github.com/skhabiri/ML-DeepLearning/blob/main/module1-rnn-and-lstm/ann_rnn_lstm-431.ipynb)
    - Trained a Recurrent Neural Networks to generate text in Shakespeare's writing style, given a random prompt. [github](https://github.com/skhabiri/ML-DeepLearning/blob/main/module1-rnn-and-lstm/ann_rnn_lstm-431a.ipynb)

### Convolutional Neural Network (CNN)
  - *Transfer Learning and Image classification:*
    - Used pretrained ResNet50 for unsupervised image classification. [github](https://github.com/skhabiri/ML-DeepLearning/blob/main/module2-convolutional-neural-networks/cnn-432.ipynb)
    - Trained a 2D convolutional neural network for multiclass classification on cifar10 dataset. Achieved 0.91 prediction accuracy. [github](https://github.com/skhabiri/ML-DeepLearning/blob/main/module2-convolutional-neural-networks/cnn-432.ipynb)
    - Trained a neural network consists of customized top layers added to the ResNet50 image processing layers for a binary classification, which achieved a val_accuracy of 0.84. [github](https://github.com/skhabiri/ML-DeepLearning/tree/main/module2-convolutional-neural-networks)
    - Built and trained a 9 layer customized CNN on an augmented dataset of mountains and forests for binary classification  Achieved accuracy score of 0.87. [github](https://github.com/skhabiri/ML-DeepLearning/blob/main/module2-convolutional-neural-networks/cnn-432a.ipynb)

### Autoencoders
  - *Unsupervised dimensions reduction:*
    - Trained an autoencoder consists of fully connected layers to reconstruct an image after dimension reduction using quickdraw10 dataset. [github](https://github.com/skhabiri/ML-DeepLearning/tree/main/module3-autoencoders)
  - *Unsupervised information retrieval:*
    - Trained an Conv2D autoencoder on Quickdraw dataset to perform a reverse image search to find similar images from a database. [github](https://github.com/skhabiri/ML-DeepLearning/blob/main/module3-autoencoders/autoencoder-433.ipynb)

### Data Science Cloud Deployment
  - *Scikit-learn Helper Package:*
    - Published a scikit-learn helper package to fit and get various metrics for a multi class classifier. Dependencies were managed by pipenv. Tested the package on different OS using Docker. Created test cases for the package using unittest python module. [package](https://pypi.org/project/skestimate/), [github](https://github.com/skhabiri/EstimatorPkg)
  - *A full-stack Twitter web App backed by machine learning model:*
    - Built a full-stack web app to predict the author of a given tweet using Flask framework. Leveraged tweepy and sqlalchemy libraries to pull tweet data from Twitter, Trained a logistic regression model on tweets and deployed it on Heroku. [github](https://github.com/skhabiri/HypoTweet), [app](https://hypotweet.herokuapp.com/), [blog post](https://skhabiri.com/2020-09-16-A_Full_Stack_Machine_Learning_Web_App_For_Twitter_Using_Flask_Framework/)
  - *A Data Science API for Spotify:*
    - Built a data science API using FastAPI to deploy a song recommendation engine for Spotify and provided multiple endpoints to interact with a frontend JavaScript web app. One of the data science API endpoints would connect to spotipy, a spotify python api, to pull data and run queries on Spotify database.[github](https://github.com/skhabiri/FastAPI-Spotify), [app](https://fastapi-spotify.herokuapp.com/), [blog post](https://skhabiri.com/2020-08-17-A-Data-Science-API-For-Spotify-Web-Applications/)

### Semi Supervised Learning
  - *Imbalanced Classification:*
    - Predicted technical review outcome for an imbalance dataset of survey data from Bridges to Prosperity initiative. Developed a data science API to deploy the machine learning model and host it on AWS Elastic Beanstalk for the final product. RandomForestClassifier was used to train the model. The project environment was containerized with Docker to ensure reproducibility. [github](https://github.com/skhabiri/Bridges2Prosperity-ML-FastAPI), [blog post](https://skhabiri.com/2020-11-18-Classification-of-Imbalanced-Dataset-provided-by-Bridges-to-Prosperity-(B2P)-and-FastAPI-Framework-deployment-to-AWS-Elastic-Beanstalk/), [app](https://b2p.skhabiri.com/)

### Regression Analysis for Supervised Machine Learning
  - *Linear Regression:*
    - Predicted the sale/rent price from New York home sale and home rent data. [github repo](https://github.com/skhabiri/PredictiveModeling-LinearModels-u2s1/tree/master/Regression-m1)
    - Modeled the relationship between “Growth in Personal Incomes” and “US Military Fatalities” using the Bread and Peace (Voting in Postwar US Presidential Elections) dataset [github repo](https://github.com/skhabiri/PredictiveModeling-LinearModels-u2s1/tree/master/Regression-m2)
  - *Ridge Regression:*
    - Identified the correlation factor between each feature and target in NYC apartment rental data. Used Ridge regressor and RidgeCV to regularize the coefficient values. [github repo1](https://github.com/skhabiri/PredictiveModeling-LinearModels-u2s1/tree/master/RidgeRegression-m3), [repo2](https://github.com/skhabiri/PredictiveModeling-TreeBasedModels-u2s2/tree/master/CrossValidation-m3)
  - *Logistic Regression:*
    - Predicted passenger survival from Titanic dataset with an accuracy of 0.81. [github repo](https://github.com/skhabiri/PredictiveModeling-LinearModels-u2s1/blob/master/LogisticRegression-m4/logisticregress-214.ipynb)
    - Trained a binary classifier to predict “Great” burritos from a dataset of 400+ burrito recipes and reviews. [dataset](https://srcole.github.io/100burritos/), [github repo](https://github.com/skhabiri/PredictiveModeling-LinearModels-u2s1/blob/master/LogisticRegression-m4/logisticregress-214a.ipynb)

### Feature Selection and Model Interpretation
  - *Permutation Importance for Tree based classifiers:*
    - Identified the most relevant features to predict the status of a water pump using Permutation Importance from eli5 in Tanzania water pump dataset. Trained and cross validated xgboost, gradient boost and random forest, Achieved an accuracy score of 0.81. [github repo](https://github.com/skhabiri/PredictiveModeling-AppliedModeling-u2s3/tree/master/PermutationBoosting-m3)
  - *SHapley Additive exPlanations:*
    - Predicted the interest rate using Lending club dataset, achieving an R2 score of 0.25 with an OrdinalEncoder() and XGBRegressor. Interpreted the model using partial dependence plot and SHapely for additive explanation. [github repo](https://github.com/skhabiri/PredictiveModeling-AppliedModeling-u2s3/tree/master/ModelInterpretation-m4)
  - *Classifier Dashboard with Plotly:*
    - Trained and tuned multiple classifiers to predict the forest cover type using the forest-cover dataset. Created and deployed a web app by Plotly dash on heroku.[github repo](https://github.com/skhabiri/PredictiveModeling-CoverType-u2build), [blog post](https://skhabiri.com/2020-07-28-A-Comparison-of-Supervised-Multi-class-Classification-Methods-for-the-Prediction-of-Forest-Cover-Types/), [web app](https://predictivemodeling-covertype.herokuapp.com/)

### Data Pipeline (ETL)
  - *Sqlite to PostgreSQL:*
    - Built an ETL to data pipeline a database of role play characters in sqlite into a postgreSQL database hosted on ElephantSQL. [github](https://github.com/skhabiri/SQL-Databases-u3s2/tree/master/postgresql-u3s2m2)
  - *csv to PostgreSQL:*
    - Pipelined a csv dataset of Titanic into a local sqlite database and from there into a postgreSQL hosted on the cloud. [github](https://github.com/skhabiri/SQL-Databases-u3s2/blob/master/postgresql-u3s2m2/insert_titanic.py)
  - *Sqlite to MongoDB:*
    - Role play character sqlite database was ETL’d into a no schema MongoDB using pymongo library. [github](https://github.com/skhabiri/SQL-Databases-u3s2/tree/master/mongodb-u3s2m3)

### Statistical Analysis
  - *Inferential Statistics:*
    - Applied independent t-test to analyze house of representative votes on various bills across party lines [data](https://archive.ics.uci.edu/ml/datasets/congressional+voting+records)
    - Applied 1-sample t-test to test null hypothesis for auto mpg [dataset](https://archive.ics.uci.edu/ml/datasets/auto+mpg)
    - Used Chi^2 test of independence on [Student Performance Data Set](https://archive.ics.uci.edu/ml/datasets/Student+Performance) to examine the level of association between various variables.
    - Determined confidence interval of respondents average age for different sample sizes of data in Census Income [dataset](https://archive.ics.uci.edu/ml/datasets/adult) using scipy.stats module.
    

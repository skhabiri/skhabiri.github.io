# Recent Applied Machine Learning and Data Science Projects
* **Natural Language Processing (NLP)**
  - Tokenization and visualization of customer reviews: Performed frequency based and w2v tokenization on review texts of “Datafiniti Amazon consumer reviews” and “Yelp coffee shop reviews” datasets separately. The tokens were statistically trimmed and visualized by Squarify library. [github]
  - Text Vectorization and similarity search: Vectorized a dataset of 401 articles from BBC website as a Bag of Words, and performed similarity search on the queried articles. [github] 
Performed similarity search on a dataset of job listing descriptions with html tags. Worked with Beautiful Soup, a Python library to pull data out of HTML files for data wrangling purposes. [github]
  - Document Classification
Performed binary classification on a newsgroups dataset with two categories of “alt.atheism” and “talk.religion.misc”, using Latent Semantic Indexing (LSI).  After hyper-parameter tuning an accuracy of 0.91 was achieved. [github]
Performed document classification on whiskey review dataset with three target label classes. [github]
Topic modeling and sentiment analysis
Used Gensim library to create a dictionary of words and tokenize reviews in an IMBD reviews dataset. Derived topic composition of each review and performed sentiment analysis for each primary topic. Visualized the topics using pyLDAvis library. [github]
Performed topic modeling on three categories of newsgroup dataset. Topic perplexity and topic coherence in applied LDA models were evaluated. [github]
A Subreddit Recommendation Engine Using NLP techniques
Built a recommendation engine that takes a text and recommends a subreddit category based on the content. Praw, a reddit api for python was utilized to pull data from reddit into a local SQL database. Achieved 0.53 accuracy for 44 subreddit categories. [github, blog post]

Feedforward Artificial Neural Networks (Perceptron)
Supervised learning using artificial neural networks
Performed hyper-parameter tuning on MNIST dataset. Achieved validation accuracy of 0.95. [github]
Trained a feed forward NN on Quickdraw dataset containing 100K drawing in ten classes. Applied common techniques to avoid overfitting. Achieved accuracy score of 0.82. [github, blog post]

Recurrent Neural Network (RNN)
Sentiment Classification 
Trained a RNN consists of Keras Embedding and LSTM layers to classify review sentiments for an IMBD Reviews dataset. Resulting in accuracy score of 0.83. [github]
Text generation
Trained a RNN on a dataset containing news articles to generate text given a seed phrase. [github]
Trained a Recurrent Neural Networks to generate text in Shakespeare's writing style, given a random prompt. [github]

Convolutional Neural Network (CNN)
Transfer Learning and Image classification
Used pretrained ResNet50 for unsupervised image classification. [github]
Trained a 2D convolutional neural network for multiclass classification on cifar10 dataset. Achieved 0.91 prediction accuracy. [github]
Trained a neural network consists of customized top layers added to the ResNet50 image processing layers for a binary classification, which achieved a val_accuracy of 0.84. [github]
Built and trained a 9 layer customized CNN on an augmented dataset of mountains and forests for binary classification  Achieved accuracy score of 0.87. [github]

Autoencoders
Unsupervised dimensions reduction
Trained an autoencoder consists of fully connected layers to reconstruct an image after dimension reduction using quickdraw10 dataset. [github]
Unsupervised information retrieval
Trained an Conv2D autoencoder on Quickdraw dataset to perform a reverse image search to find similar images from a database. [github]

Data Science Cloud Deployment
Scikit-learn Helper Package
Published a scikit-learn helper package to fit and get various metrics for a multi class classifier. Dependencies were managed by pipenv. Tested the package on different OS using Docker. Created test cases for the package using unittest python module. [package, github]
A full-stack Twitter web App backed by machine learning model
Built a full-stack web app to predict the author of a given tweet using Flask framework. Leveraged tweepy and sqlalchemy libraries to pull tweet data from Twitter, Trained a logistic regression model on tweets and deployed it on Heroku. [github, app, blog post]
A Data Science API for Spotify
Built a data science API using FastAPI to deploy a song recommendation engine for Spotify and provided multiple endpoints to interact with a frontend JavaScript web app. One of the data science API endpoints would connect to spotipy, a spotify python api, to pull data and run queries on Spotify database.[github, app, blog post]

Semi Supervised Learning
Imbalanced Classification
Predicted technical review outcome for an imbalance dataset of survey data from Bridges to Prosperity initiative. Developed a data science API to deploy the machine learning model and host it on AWS Elastic Beanstalk for the final product. RandomForestClassifier was used to train the model. The project environment was containerized with Docker to ensure reproducibility. [github, blog post, app]

Regression Analysis for Supervised Machine Learning
Linear Regression
Predicted the sale/rent price from New York home sale and home rent data. [github repo]
Modeled the relationship between “Growth in Personal Incomes” and “US Military Fatalities” using the Bread and Peace (Voting in Postwar US Presidential Elections) dataset [github repo]
Ridge Regression 
Identified the correlation factor between each feature and target in NYC apartment rental data. Used Ridge regressor and RidgeCV to regularize the coefficient values. [github repo1, repo2]
Logistic Regression
Predicted passenger survival from Titanic dataset with an accuracy of 0.81. [github repo]
Trained a binary classifier to predict “Great” burritos from a dataset of 400+ burrito recipes and reviews. [dataset, github repo]

Feature Selection and Model Interpretation
Permutation Importance for Tree based classifiers
Identified the most relevant features to predict the status of a water pump using Permutation Importance from eli5 in Tanzania water pump dataset. Trained and cross validated xgboost, gradient boost and random forest, Achieved an accuracy score of 0.81. [github repo]
SHapley Additive exPlanations
Predicted the interest rate using Lending club dataset, achieving an R2 score of 0.25 with an OrdinalEncoder() and XGBRegressor. Interpreted the model using partial dependence plot and SHapely for additive explanation. [github repo]
Machine Learning App
Trained and tuned multiple classifiers to predict the forest cover type using the forest-cover dataset. Created and deployed a web app by Plotly dash on heroku.[github repo, blog post, web app]

Data Pipeline (ETL)
Sqlite to PostgreSQL
Built an ETL to data pipeline a database of role play characters in sqlite into a postgreSQL database hosted on ElephantSQL. [github]
csv to PostgreSQL
Pipelined a csv dataset of Titanic into a local sqlite database and from there into a postgreSQL hosted on the cloud. [github]
Sqlite to MongoDB
Role play character sqlite database was ETL’d into a no schema MongoDB using pymongo library. [github]

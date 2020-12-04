# #howistwitterfeeling
<img src="jupyter notebook/image/morning-brew-V6CdmV277nY-unsplash.jpg">

## Background and Problem Statement
Twitter is a platform used by millions of users everyday to share their unfiltered opinion, follow public figures and the news. While it is a quick and convenient source of information, it is also known as a place of dispute and disagreements. Given the massive quantity of contents posted on the platform regarding different topics everyday, it could be difficult to keep track of how users are actually perceiving a specific topic or issue. Like many other social media data, Twitter data is unstructured, hence another challenge working with them, would be how should automatize the data cleaning and preprocessing process.

Twitter sentiment analysis has been a really popular topic in the data science community, some researchers have been exploring whether or not the existing lexical resources and features are able to capture information about the informal language used in microblogging platforms like Twitter. While others are interested in learning and capturing public sentiments regarding a specific topic, for instance, the US Election and Climate Change. 

These researches inspired me to build my own text analytic model, with the intention of capturing Twitter users' real-time opinions regarding a certain topic or issue, using sentiment analysis.


## Data Acquisition
To train my model, I have chosen to use the Sentiment140 datasets with 1.6 million tweets from Kaggle (https://www.kaggle.com/kazanova/sentiment140). This dataset contains 1,600,000 tweets extracted using the twitter api. The tweets have been annotated (0 = negative, 4 = positive) and they can be used to detect sentiment .
It contains the following 6 fields:
- target: the polarity of the tweet (0 = negative, 4 = positive)
- ids: The id of the tweet ( 2087)
- date: the date of the tweet (Sat May 16 23:58:44 UTC 2009)
- flag: The query (lyx). If there is no query, then this value is NO_QUERY.
- user: the user that tweeted (robotickilldozr)
- text: the text of the tweet (Lyx is cool)
<img src="jupyter notebook/image/raw_data.png">

Then, I have chosen to use Tweepy, an open source Python package for accessing Twitter API, to livestream my tweets. The reason why I want to use Tweepy is because otherwise I might have to deal with low-level details like HTTP requests, data serialization, rate limits, etc. So Tweepy definitely makes the process a lot easier. The tweets are pulled using the tweepy.Cursor() method, which enables me to specify:
text_query : the Twitter topic I’m interested in
coordinates : the location of the tweets are from
language : written language of tweet
since_date and until_date : the time period I want the retrieved tweets to fall under
max_tweets : the number of tweets I want to retrieve

Then, the date and contents of the retrieved tweets are placed into a dataframe, where I would apply further feature engineering upon.

<img src="jupyter notebook/image/twitter_data.png">


## Data Preprocess and Vectorization
After conducting an initial EDA on my Kaggle dataset, I have found the following information about each columns to facilitate me in the data cleaning and preprocessing steps:
- There appears to be duplicated tweet_id in the dataset, while tweet_id should be an unique identifier for each tweet. After further investigation, I have found that there are duplicates of the same entries labelled with contradicting sentiment. Looking at the contents of these entries, it appears that a lot of them contain satirical or neutral languages.
Solution: Remove these 1,685 entries (0.1%) from the dataset.

- From the date column, it appears that all of the tweets are documented using the Pacific Time timezone, and they are posted between April 2009 to June 2009.
Solution: In the feature engineering process, do not include time zone and year as new numerical features.

- It appears all of the rows in the flag column contain the value of NO_QUERY, hence this column will not bring any information to our model.
Solution: Remove the flag column entirely from the dataset.

With these information found in the EDA, I have engineered a few additional date-time and numerical features in addition to the text data, including hour,day, month,weekday columns from the date column and hashtags, hashtag counts, mentions, mention counts, tagged URL and URL counts from the text column. I have chosen these new features, as I thought users’ emotions may differ throughout the hour, days and months, hence the date-time features could potentially offer new information about the sentiments. The number of hashtags, mentions and URLs may also reflect the nature of the tweet, for instance, a tweet with multiple URLs might more likely be an advertisement, hence the sentiment might be more positive.

Then, I preprocessed the text column, with the following steps:
1. Convert all characters to lowercase
2. Remove punctuations
3. Remove stop words with the english stop word dictionary from NLTK library
4. Lemmatize the words with NLTK’s POS tags

I have chosen to use the POS tags, as by default, the lemmatizer takes in an input string and tries to lemmatize it, so if you pass in a word, it would lemmatize it treating it as a noun. Hence, To make the lemmatization better and context dependent, we would need to find out the POS tag and pass it on to the lemmatizer. 

Lastly, I vectorize the preprocessed text column, with the following conditions using the TFIDFvectorizer, max_feature, min_df and max_df, stop_words, token_patterns, strip_accents.


## Data Modelling
There are two approaches I used for data modelling, incremental learning, which trains the model on the entire datasets (1.6 million entries)  by partially fitting the batches with the SGDClassifier. And our typical ML approach with reduced data, which only trains the model on a sample (100,000 entries) of the entire dataset.

Machine Learning with Reduced Dataset:
With the reduced dataset, there are four models I want to attempt to train my model on:
- LogisticRegression
- LinearSVC
- DecisionTreeClassifier
- KNeighborsClassifier

With an initial model selection, comparing the train accuracy, test accuracy and CV training accuracy:
<img src="jupyter notebook/image/reduced_accuracy.png">

I have chosen to proceed with the LinearSVC and LogisticRegression for further modelling. These are the hyperparameters I have chosen to tune in the two models:
- LogisticRegression: Penalty (to compare regularization techniques) and regularization parameter C
- LinearSVC: class_weight (due to the high false negative and low recall rate) and regularization parameter C (for SVM structural risk minimization)

After tuning the respective hyperparameters, the best model appears to be the LinearSVC model with C = 1 and class_weight = {-1: 1, 1: 5}, with a train accuracy of **78.56%** and test accuracy of **74.84%** and a much better recall rate (0.732). Hence, this model is pickled for sentiment analysis of the real-time tweets.

Model with Incremental Learning:
After deriving the appropriate model and hyperparameters for the reduced dataset, I will apply them in a model using incremental learning as well, to train my model based on more data. Given the restriction in the memory on my local computer, incremental learning techniques can be used where data is processed in parts(subsets of the data are considered at any given point in time) and the result is then combined to save memory.I have used the SGDClassifier, which is capable of implementing regularized linear models, like the logistics and SVM, with stochastic gradient descent learning, as the gradient of loss is estimated each sample at a time and the model is updated along the way.

After fitting the entire dataset, I only received a test accuracy of **50.09%** from the model. Hence, for now, for the prediction of sentiment, I will use the model fitted with the reduced dataset. However, this also means that I have underutilized my dataset, leading to some degrees of loss of information. So I will definitely be looking into how I can optimize the performance of my models based on incremental learning.

Python Packages Used:
- Numpy
- Pandas
- Matplotlib
- Scikit-Learn
- NLTK
- Tweepy (Twitter API)

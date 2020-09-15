# Import the relevant packages
import os
import tweepy
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

import pickle
from sklearn.feature_extraction.text import TfidfTransformer

# define your parameters
text_query = "Coronavirus"
coordinates = '43.651070,-79.347015,50mi'
language = 'en'
result_type = 'recent'
since_date = '2020-09-06'
until_date = '2020-09-13'
max_tweets = 10000

# define your keys
consumer_key = '2YCaHB1rnU7I7U8BuDJVqPGP2'
consumer_secret = 'UJR0oFVc6JzaoWC6J2K2n3cMEfdAZS6nhJtHTGeHBnehrFVPZw'
access_token = '1280193789756309511-b6F7ZCckvK3crRh7EzhfKk0sIJBYXQ'
access_token_secret = '1I3YLccFnFoGzP0ekSWGLgXdUVBFHyVhnmvOs2ZQWX1XX'

auth = tweepy.OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token(access_token,access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)
 
# Creation of query method using parameters
tweets = tweepy.Cursor(api.search,
                       q = text_query,
                       geocode = coordinates,
                       lang=language,
                       result_type = result_type,
                       since = since_date,
                       until = until_date,
                       count = 100).items(max_tweets)

# List comprehension pulling chosen tweet information from tweets iterable object
# Add or remove tweet information you want in the below list comprehension
tweets_list = [[tweet.created_at,tweet.text] for tweet in tweets]

# Creation of dataframe from tweets_list
# Did not include column names to simplify code 
tweets_df = pd.DataFrame(tweets_list,columns=['date','text'])

# Feature Engineering
# Hours
tweets_df['hour'] = [dt.hour for dt in tweets_df['date'].astype(object)]
# Days
tweets_df['day'] = [dt.day for dt in tweets_df['date'].astype(object)]
# Month
tweets_df['month'] = [dt.month for dt in tweets_df['date'].astype(object)]
# Weekday
tweets_df['dayofweek'] = [dt.dayofweek for dt in tweets_df['date'].astype(object)]
# Delete date column
tweets_df = tweets_df.drop(['date'],axis=1)
# Retrieve the hashtags and add the column to the dataset
hashtags = []
for tweet in tweets_df['text']:
    hashtags.append([i  for i in tweet.split() if i.startswith("#") ])
tweets_df['hashtags'] = hashtags
# Find number of hashtags in each tweet
hashtag_counts = []
for hashtag in hashtags:
    hashtag_counts.append(len(hashtag))
tweets_df['hashtag_counts'] = hashtag_counts
# Retrieve the user names and add the column to the dataset
users = []
for tweet in tweets_df['text']:
    users.append([i for i in tweet.split() if i.startswith("@") ])
tweets_df['users'] = users
# Find number of tagged users in each tweet
user_counts = []
for user in users:
    user_counts.append(len(user))
tweets_df['user_counts'] = user_counts
# Drop users column
tweets_df = tweets_df.drop(['users'],axis=1)
# Retrieve the URLs from the tweets
from urlextract import URLExtract
extractor = URLExtract()
urls = []
for i in range(len(tweets_df)):
    urls.append(extractor.find_urls(tweets_df['text'][i]))
tweets_df['urls'] = urls
# Find number of urls in each tweet
url_counts = []
for url in tweets_df['urls']:
    url_counts.append(len(url))
tweets_df['url_counts'] = url_counts
# Drop urls column
tweets_df = tweets_df.drop(['urls'],axis=1)
# Remove excessive information from text Column
import re
# Creating a function called clean, that removes all hyperlink, hashtags and mentions
def clean(x):
    x = re.sub(r"^RT[\s]+", "", x)
    x = re.sub(r"https?:\/\/.*[\r\n]*", "", x)
    #x = re.sub('[^ ]+\.[^ ]+','',x)
    x = re.sub(r"#","", x)
    x = re.sub(r"@[A-Za-z0–9]+","", x)
    return x  
# Apply the clean function to text column
tweets_df['text'] = tweets_df['text'].apply(clean)

# Vectorization
# Load features from training dataset
transformer = TfidfTransformer()
loaded_features = pickle.load(open("pickle/feature.pkl", "rb"))

# Vectorize the text column
X_text = tweets_df['text'].astype(str)
tfidfconverter = TfidfVectorizer(max_features=10000, 
                                 min_df=5, max_df=0.7, 
                                 stop_words=stopwords.words('english'),
                                 token_pattern=r'\b[^\d\W]+\b',
                                 strip_accents = "ascii",
                                 vocabulary = loaded_features)

# Convert the features in test set to train set
X_text = transformer.fit_transform(tfidfconverter.fit_transform(X_text))
X_text = pd.DataFrame(columns=tfidfconverter.get_feature_names(),data=X_text.toarray())
# Retrieve the numerical columns
X_num = tweets_df.drop(['text','hashtags','topic'],axis=1)
# Concatenate the test dataset
X_sample = pd.concat([X_num,X_text],axis=1).astype('int64')

# load the model from disk
filename = 'sgd_model.sav'
sgd_incremental_model = pickle.load(open(filename, 'rb'))




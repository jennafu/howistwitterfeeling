# Web app packages
import streamlit as st
import tweepy

# Import the relevant packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from datetime import timedelta, date

import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

import emoji
import regex

import matplotlib.pyplot as pPlot
from wordcloud import WordCloud

st.title('#How is Twitter Feeling About...')

option = st.selectbox(
    'Which Twitter topic would you like to explore? ',
   ('Raptors','Coronavirus', 'US Election', 'Back to School'))
    

st.header('Topic sentiment in the past 7 days:')
st.text('Sentiment Score on a scale of -1 to 1' )
st.text('(with -1 being the most negative and 1 being the most positive)' )

# define your keys
consumer_key = '2YCaHB1rnU7I7U8BuDJVqPGP2'
consumer_secret = 'UJR0oFVc6JzaoWC6J2K2n3cMEfdAZS6nhJtHTGeHBnehrFVPZw'
access_token = '1280193789756309511-b6F7ZCckvK3crRh7EzhfKk0sIJBYXQ'
access_token_secret = '1I3YLccFnFoGzP0ekSWGLgXdUVBFHyVhnmvOs2ZQWX1XX'

auth = tweepy.OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token(access_token,access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)

# Derive the date of last 7 days
def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)

start_date = (date.today() + timedelta(days=-6))
end_date = (date.today() + timedelta(days=1))
date_list = []
for single_date in daterange(start_date, end_date):
    date_list.append(single_date)

# Retrive topic-specific tweets
tweets_list_date = list()
tweets_list_text = list()

for date_ in date_list:
    text_query = option
    coordinates = '43.651070,-79.347015,50mi'
    language = 'en'
    result_type = 'recent'
    since_date = date_
    until_date = (date_ + timedelta(days=1))
    max_tweets = 1000
    
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
    for tweet in tweets:
        tweets_list_date.append(tweet.created_at)
        tweets_list_text.append(tweet.text)
 
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
# Add an independent column date
date = tweets_df['date']
date = pd.to_datetime(date).dt.date

# Creation of dataframe from tweets_list
# Did not include column names to simplify code 
tweets_df = pd.DataFrame({'date' : tweets_list_date,'text' : tweets_list_text},columns=['date','text'])
# Add an independent column date
date = tweets_df['date']
date = pd.to_datetime(date).dt.date

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
X_num = tweets_df.drop(['text','hashtags'],axis=1)
# Concatenate the test dataset
X_sample = pd.concat([X_num,X_text],axis=1).astype('int64')

# Load trained model
filename = 'pickle/svc_model.sav'
sgd_model = pickle.load(open(filename, 'rb'))

# Prediction
y_sample = sgd_model.predict(X_sample)
y_prediction = pd.DataFrame(y_sample,columns = ["prediction"])
y_prediction = pd.concat([date,y_prediction],axis=1)
y_prediction_table = y_prediction.groupby('date').mean()

# Create figure and plot space
fig, ax = plt.subplots(figsize=(12, 6))
# Add x-axis and y-axis
ax.plot(y_prediction_table.index,
           y_prediction_table.prediction,
       color='blue')
# Set title and labels for axes
ax.set(xlabel="Date",
       ylabel="Sentiment Score",)

# Set limit for y scale
ax.set_ylim(y_prediction_table.prediction.min()-0.01,y_prediction_table.prediction.max()+0.03)

# Define the date format
date_form = DateFormatter("%m-%d")
ax.xaxis.set_major_formatter(date_form)
st.pyplot()

st.subheader('Top 10 Emojis Used:')
 
def split_count(text):

    '''
    Return the emojis found in the twitter texts associated with the topic
    Input: text column from dataset
    Output: A list of emojis found in each rows
    '''
    emoji_list = []
    data = regex.findall(r'\X', text)
    for word in data:
        if any(char in emoji.UNICODE_EMOJI for char in word):
            emoji_list.append(word)

    return emoji_list

# Return the emojis found in the twitter texts
emoji_rows = tweets_df['text'].apply(split_count)

# Return a flattened list of emojis from the topic
emoji_list = []
for sublist in emoji_rows:
    for item in sublist:
        emoji_list.append(item)
        
emoji_count = [[x,emoji_list.count(x)] for x in set(emoji_list)]

emoji_df = pd.DataFrame(emoji_count,columns=['Emoji','Count']).sort_values('Count',ascending=False)
st.dataframe(emoji_df.head(10))

st.subheader('Associated Hashtags Word Cloud:')

# Return a flattened list of emojis from the topic
hashtag_list = []
for sublist in tweets_df['hashtags']:
    for item in sublist:
        hashtag_list.append(item)
hashtag_str = ' '.join(hashtag for hashtag in hashtag_list)

# Create and generate a word cloud image:
wordcloud = WordCloud().generate(hashtag_str)
# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
st.pyplot()
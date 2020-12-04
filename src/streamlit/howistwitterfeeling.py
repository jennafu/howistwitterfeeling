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
from textblob import TextBlob

import emoji
import regex

import matplotlib.pyplot as pPlot
from wordcloud import WordCloud

from PIL import Image
image = Image.open('src/streamlit/image/morning-brew-V6CdmV277nY-unsplash.jpg')
st.image(image,use_column_width=True)
# st.title('How is Twitter Feeling About...')

st.title('#How is Twitter Feeling About...')

st.text('Welcome to the app! This app serves to provide real-time insights to what Twitter users \n'
'are feeling about a certain topic or issue. Enter the topic you are interested in below:')

option = st.text_input("Which topic would you like to explore?", 'Coronavirus')
    

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

with st.spinner('Scraping Twitter Data....'):
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
        tweets = tweepy.Cursor(api.search,q = text_query,geocode = coordinates,lang=language,
                           result_type = result_type,since = since_date,until = until_date,
                           count = 100).items(max_tweets)
                           
        # List comprehension pulling chosen tweet information from tweets iterable object
        # # Add or remove tweet information you want in the below list comprehension
        for tweet in tweets:
            tweets_list_date.append(tweet.created_at)
            tweets_list_text.append(tweet.text)
            
    # List comprehension pulling chosen tweet information from tweets iterable object
    # # Add or remove tweet information you want in the below list comprehension
    tweets_list = [[tweet.created_at,tweet.text] for tweet in tweets]
    
    # Creation of dataframe from tweets_list
    tweets_df = pd.DataFrame({'date' : tweets_list_date,'text' : tweets_list_text},columns=['date','text'])
st.success(f'Scrapped {tweets_df.shape[0]} tweets in the GTA region from Twitter.')

# Add an independent column date
date = tweets_df['date']
date = pd.to_datetime(date).dt.date

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
loaded_features = pickle.load(open("src/streamlit/pickle/reduced_new_feature.pkl", "rb"))

# Vectorize the text column
X_text = tweets_df['text'].astype(str)
tfidfconverter = TfidfVectorizer(max_features=10000, 
                                 ngram_range=(1,2),
                                 min_df=0.0001, max_df=0.5, 
                                 stop_words=stopwords.words('english'),
                                 token_pattern=r'\b[^\d\W]+\b',
                                 strip_accents = "ascii",
                                 vocabulary = loaded_features)

# Convert the features in test set to train set
X_text = transformer.fit_transform(tfidfconverter.fit_transform(X_text))
X_sample = pd.DataFrame(columns=tfidfconverter.get_feature_names(),data=X_text.toarray())

# Load trained model
filename = 'pickle/svc_new_model.sav'
sgd_model = pickle.load(open(filename, 'rb'))

# Prediction
y_sample = sgd_model.predict(X_sample)
y_prediction = pd.DataFrame(y_sample,columns = ["prediction"])
y_prediction = pd.concat([date,y_prediction],axis=1)
# Get tweet sentiments using TextBlob
textblob_y = list()
for text in tweets_df['text']:
    testimonial = TextBlob(text)
    textblob_y.append(testimonial.sentiment.polarity)

# Function to find index of neutral tweets
def get_index_positions(list_of_elems, element):
    ''' Returns the indexes of all occurrences of give element in
    the list- listOfElements '''
    index_pos_list = []
    index_pos = 0
    while True:
        try:
            # Search for item in list from indexPos to the end of list
            index_pos = list_of_elems.index(element, index_pos)
            # Add the index position in list
            index_pos_list.append(index_pos)
            index_pos += 1
        except ValueError as e:
            break
    return index_pos_list
# Label tweets with neutral sentiments with the value of 0.5
y_prediction['prediction'][get_index_positions(textblob_y,0.0)] = 0.5
# Find the average sentiments in the last 7 days
y_prediction_table = y_prediction.groupby('date').mean()

# Create figure and plot space
import plotly.graph_objects as go

st.text('     ')

st.text('The chart below returns the average sentiment scores of the last 7 days, \n'
'from a scale of 0 to 1. The scores are calculated by the averaging the prediction \n'
'scores (0 or 1) on a single day.')

fig = go.Figure(data=go.Scatter(x=y_prediction_table.index, y=y_prediction_table.prediction))

fig.update_layout(
    title={
        'text': "Changes in Twitter Sentiments over Last 7 Days",
        'y':0.9,
        'x':0.45,
        'xanchor': 'center',
        'yanchor': 'top'},
        yaxis_title="😠 <== Average Sentiment Scores ==> 😄",
        xaxis_title="Date")

st.plotly_chart(fig)
 
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

st.text('     ')

st.text('The chart below finds and ranks the top 10 most frequently appearing emojis \n'
'in the tweets we have scrapped from Twitter.')

# Convert the list into a dataframe
emoji_df = pd.DataFrame(emoji_count,columns=['Emoji','Count']).sort_values('Count',ascending=False)

trace1 = {
  "type": "bar", 
  "x": emoji_df['Emoji'][0:11], 
  "y": emoji_df['Count'][0:11]
}
data = go.Data([trace1])

fig = go.Figure(data=data)

fig.update_layout(
    title={
        'text': "Top 10 Emojis Associated with Topic",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
        yaxis_title="Emoji Count",
        xaxis_title="Emoji")

st.plotly_chart(fig)

# Merge sentiment and text columns into dataframe
text_sent = pd.concat([y_prediction['prediction'],tweets_df['text']],axis=1)

st.text('     ')

st.text('The word cloud below represents the words or phrases from tweets \n'
'associated with positive and negative senitments accordingly.')

# Create the word cloud
bird = np.array(Image.open('image/twitter_mask.png'))
fig, (ax2, ax3) = plt.subplots(1, 2, figsize=[30, 15])
wordcloud2 = WordCloud( background_color='white',mask=bird,colormap="Reds",
                        width=600,stopwords=option,
                        height=400).generate(" ".join(text_sent[text_sent['prediction']==0]['text']))
ax2.imshow(wordcloud2)
ax2.axis('off')
ax2.set_title('Negative Sentiment',fontsize=35)

wordcloud3 = WordCloud( background_color='white',mask=bird,colormap="Greens",
                        width=600,stopwords=option,
                        height=400).generate(" ".join(text_sent[text_sent['prediction']==1]['text']))
ax3.imshow(wordcloud3)
ax3.axis('off')
ax3.set_title('Positive Sentiment',fontsize=35)
plt.show()
st.pyplot()

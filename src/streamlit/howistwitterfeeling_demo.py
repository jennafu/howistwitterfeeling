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

from PIL import Image
image = Image.open('image/morning-brew-V6CdmV277nY-unsplash.jpg')
st.image(image,use_column_width=True)
# st.title('How is Twitter Feeling About...')

option = st.text_input("Which topic would you like to explore?", 'Coronavirus')

st.header('Topic sentiment in the past 7 days:')
st.text('Sentiment Score on a scale of -1 to 1' )
st.text('(with -1 being the most negative and 1 being the most positive)' )

# Import the relevant datasets for demo
if option == "Raptors":
    y_prediction_table = pd.read_csv('demo_data/raptor_sentiment.csv')
    tweets_df = pd.read_csv('demo_data/raptor_tweets.csv',index_col=0)
    hashtag_str = open('demo_data/raptor_hashtags.txt', 'r').read()
elif option == "Coronavirus":
    y_prediction_table = pd.read_csv('demo_data/covid_sentiment.csv')
    tweets_df = pd.read_csv('demo_data/covid_tweets.csv',index_col=0)
    hashtag_str = open('demo_data/covid_hashtags.txt', 'r').read()

# Create figure and plot space
fig, ax = plt.subplots(figsize=(12, 6))
# Add x-axis and y-axis
ax.plot(y_prediction_table['date'],y_prediction_table.prediction,
       color='blue')
# Set title and labels for axes
ax.set(xlabel="Date",
       ylabel="Sentiment Score",)

# Set limit for y scale
ax.set_ylim(y_prediction_table.prediction.min()-0.01,y_prediction_table.prediction.max()+0.03)

# Define the date format
#date_form = DateFormatter("%m-%d")
#ax.xaxis.set_major_formatter(date_form)
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
tweets_df['text'] = tweets_df['text'].astype(str)
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

# Create and generate a word cloud image:
wordcloud = WordCloud(collocations=False).generate(hashtag_str)
# Display the generated image:
plt.imshow(wordcloud)
plt.axis("off")
st.pyplot()
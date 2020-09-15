# Import relevant packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

train = pd.read_csv('data/feature_engineered_2.csv')
train['text']=train['text'].astype(str)

# Global Parameters
stop_words = set(stopwords.words('english'))

# Lemmatize with POS Tag
def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

    # Create a function preprocess_tweet_text
def preprocess_tweet_text(tweet):
    # Convert all characters to lower case
    tweet.lower()
    # Remove punctuations
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    # Remove stopwords
    tweet_tokens = word_tokenize(tweet)
    filtered_words = [w for w in tweet_tokens if not w in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_output = ' '.join([lemmatizer.lemmatize(w,get_wordnet_pos(w)) for w in filtered_words])
    return " ".join(filtered_words)

# Apply the preprocess_tweet_text to text column
train['text'] = train['text'].apply(preprocess_tweet_text)

# Save preprocessed dataset
train.to_csv(r'data/preprocess_data.csv', index = False)
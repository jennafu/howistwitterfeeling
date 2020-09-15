import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Retrieve urls from tweets using URLExtract
# from urlextract import URLExtract

# extractor = URLExtract()
#urls = []

# for i in range(len(train)):
    #urls.append(extractor.find_urls(train['text'][i]))

# Read in data after URL extraction
train = pd.read_csv('data/feature_engineered_1_data_partial.csv')

# Remove '][' from the `urls` column
train['urls'] = train['urls'].str[1:-1]
# Split the urls by ','
train['urls'] = train['urls'].str.split(", ")
# Find number of urls in each tweet
url_counts = []
for url in train['urls']:
    if url[0] == '':
        url_counts.append(0)
    else:
        url_counts.append(len(url))
train['url_counts'] = url_counts

# Creating a function called clean, that removes all hyperlink, hashtags, mentions and emojis
def clean(x):
    x = re.sub(r"^RT[\s]+", "", x)
    x = re.sub(r"https?:\/\/.*[\r\n]*", "", x)
    x = re.sub('[^ ]+\.[^ ]+','',x)
    x = re.sub(r"#","", x)
    x = re.sub(r"@[A-Za-z0–9]+","", x)
    return x  
# Apply the clean function to text column
train['text'] = train['text'].apply(clean)

# Remove the url, user columns from dataset and remove hastag symbols from hashtag column
train.drop(['hashtags'],axis=1,inplace=True)
train.drop(['user'],axis=1,inplace=True)
train.drop(['users'],axis=1,inplace=True)
train.drop(['urls'],axis=1,inplace=True)
train.drop(['tweet_id'],axis=1,inplace=True)

train.to_csv(r'data/feature_engineered_2_data.csv', index = False)
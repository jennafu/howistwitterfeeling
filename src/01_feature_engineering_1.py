import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Download the data file
train = pd.read_csv('data/raw_data.csv',
                       encoding = "ISO-8859-1", engine='python',header=None,
                      names=['sentiment','tweet_id','date','flag','user','text'])

# Split date column into datetime features
# Hours
train['date_hour'] = train['date'].str[11:13]
# Days
train['date_day'] = train['date'].str[8:10].astype(int)
# Month
train['date_month'] = train['date'].str[4:7]
# Weekday
train['date_weekday'] = train['date'].str[0:4]
# Convert month to numerical values
train['date_month'].replace('Apr', 4, inplace=True)
train['date_month'].replace('May', 5, inplace=True)
train['date_month'].replace('Jun', 6, inplace=True)
# Convert weekday to numerical values
train['date_weekday'].replace('Mon ', 1, inplace=True)
train['date_weekday'].replace('Tue ', 2, inplace=True)
train['date_weekday'].replace('Wed ', 3, inplace=True)
train['date_weekday'].replace('Thu ', 4, inplace=True)
train['date_weekday'].replace('Fri ', 5, inplace=True)
train['date_weekday'].replace('Sat ', 6, inplace=True)
train['date_weekday'].replace('Sun ', 7, inplace=True)
# Drop the date column
train.drop(['date'], axis=1, inplace=True)

# Retrieve features from text column
# Retrieve the hashtags and add the column to the dataset
hashtags = []
for tweet in train['text']:
    hashtags.append([i  for i in tweet.split() if i.startswith("#") ])
train['hashtags'] = hashtags
# Find number of hashtags in each tweet
hashtag_counts = []
for hashtag in hashtags:
    hashtag_counts.append(len(hashtag))
train['hashtag_counts'] = hashtag_counts
# Retrieve the user names and add the column to the dataset
users = []
for tweet in train['text']:
    users.append([i for i in tweet.split() if i.startswith("@") ])
train['users'] = users
# Find number of tagged users in each tweet
user_counts = []
for user in users:
    user_counts.append(len(user))
train['user_counts'] = user_counts

train.to_csv(r'data/feature_engineered_1_data.csv', index = False)
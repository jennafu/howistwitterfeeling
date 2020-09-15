# Import relevant packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

train = pd.read_csv('data/preprocess_data.csv')
train = train.sample(frac=1).reset_index(drop=True)

#
print('Setting up our predictor and target columns')
X = train.drop(['sentiment'],axis=1)
y = train[['sentiment']]

#
print('Vectorizing the text column')
X_text = X['text'].astype(str)
vectorizer = TfidfVectorizer(max_features=10000, 
                                 min_df=5, max_df=0.7, 
                                 stop_words=stopwords.words('english'),
                                 token_pattern=r'\b[^\d\W]+\b',
                                 strip_accents = "ascii")
X_text = vectorizer.fit_transform(X_text)
X_text = pd.DataFrame(columns=vectorizer.get_feature_names(),data=X_text.toarray())

# 
print('Saving vectorizer.vocabulary_')
pickle.dump(vectorizer.vocabulary_,open("pickle/feature.pkl","wb"))

#
print('Dropping features and only keep numerical features')
X_num = train.drop(['sentiment','text'],axis=1)

#
print('Merging the vectorized and numerical features')
# X_num split
n = 100000  #chunk row size
list_num = [X_num[i:i+n] for i in range(0,X_num.shape[0],n)]
# X_text split
n = 100000  #chunk row size
list_text = [X_text[i:i+n] for i in range(0,X_text.shape[0],n)]

#
print('Saving chunks of merged X and y datasets')
for i in range(0,16):
    X = pd.concat([list_num[i],list_text[i]],axis=1).astype('int64')
    X.to_csv(f'data/cleaned_chunk_train_data/X_train/X_{i}.csv.gz',compression='gzip')
batch_size = 100000
for i in range(0, len(y), batch_size):
    y_sample = y[i:i+batch_size]
    y_sample.to_csv(f'data/cleaned_chunk_train_data/y_train/y_{int(i/batch_size)}.csv.gz',compression='gzip')
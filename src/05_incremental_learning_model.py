# Relevant packages
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler

# Import the sample X_0 and y_0
import gzip
with gzip.open('data/X_train/X_0.csv.gz') as X:
    X = pd.read_csv(X,index_col=0)
with gzip.open('data/y_train/y_0.csv.gz') as y:
    y = pd.read_csv(y,index_col=0)

# Instantiate and fit to the train set
SGDClass = SGDClassifier(loss='log',
                         penalty = 'l2',
                         alpha = 0.001,
                         random_state=7)

# Scale the train and test sets
scaler = StandardScaler()

for i in range(0,16):
    # Import the sample X and y
    with gzip.open(f'data/X_train/X_{i}.csv.gz') as X:
        X = pd.read_csv(X,index_col=0)
    with gzip.open(f'data/y_train/y_{i}.csv.gz') as y:
        y = pd.read_csv(y,index_col=0)
        
    # Partial fit the data to scaler
    scaler.partial_fit(X)
    scaler.transform(X)
    
    # Partial fit the data
    SGDClass.partial_fit(X,y,classes=np.unique(y))
    
    # Progress
    print(f"Fitted dataset {i}")

filename = 'sgd_incremental_model.sav'
# save the model to disk
pickle.dump(SGDClass, open(filename, 'wb'))
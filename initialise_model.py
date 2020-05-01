
# Installing LIME
#!pip install LIME


import numpy as np
import pandas as pd
import joblib

df = pd.read_csv('data/train/clean_fico_data_train.csv')

testdf = pd.read_csv('data/test/clean_fico_data_test.csv')

# Dividing Dataframe into target feature (Y) and predictor features (X)
X_train = df.iloc[:, 1:24].to_numpy()
y_train = df.iloc[:, 0].to_numpy()


# Train Test Split
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

X_test = testdf.iloc[:, 1:24].to_numpy()
y_test = testdf.iloc[:, 0].to_numpy()


# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

np.save('matrices/X_train.npy', X_train)
np.save('matrices/X_test.npy', X_test)
np.save('matrices/y_test.npy', y_test)

# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=20) # n_estimators is the no. of trees in the random forest
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test) # change the string values in X?

# Algorithm Evaluation
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Score", clf.score(X_test, y_test))
# save the model to disk
filename = 'models/finalized_model.pkl'
joblib.dump(clf, filename)
 



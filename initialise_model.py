import numpy as np
import pandas as pd
import joblib

df = pd.read_csv('data/train/clean_fico_data_train.csv')
testdf = pd.read_csv('data/test/clean_fico_data_test.csv')

# Dividing Dataframe into target feature (Y) and predictor features (X)
X_train = df.iloc[:, 1:24].to_numpy()
y_train = df.iloc[:, 0].to_numpy()

X_test = testdf.iloc[:, 1:24].to_numpy()
y_test = testdf.iloc[:, 0].to_numpy()

# Feature Scaling
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)

np.save('matrices/X_train.npy', X_train)
np.save('matrices/X_test.npy', X_test)
np.save('matrices/y_test.npy', y_test)

# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=70, min_samples_leaf=5, min_samples_split=5, max_depth=15) # MODEL WITH TUNED PARAMETERS
# clf = RandomForestClassifier(n_estimators=20) # n_estimators is the no. of trees in the random forest
clf.fit(X_train, y_train)
y_pred = clf.predict_proba(X_test)
y_predf = y_pred[:,1]>0.5 # change the string values in X?
#y_pred = y_pred.astype(int)

# Algorithm Evaluation
# print("Accuracy:",metrics.accuracy_score(y_test, y_predf))
y_test = np.array(y_test, dtype=bool)
from sklearn import metrics
print(type(y_test))
print(type(y_pred))

acc = np.sum(np.logical_not(np.logical_xor(y_test, y_predf)))/len(y_test)
print("Accuracy: ", acc)
print("Score: ", clf.score(X_test, y_test))
print("Classes: ", clf.classes_)

# Save the model to disk
filename = 'models/finalized_model.pkl'
joblib.dump(clf, filename)
 



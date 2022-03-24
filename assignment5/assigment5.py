import numpy as np
#import tensorflow as tf
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pickle

with open(file="scikit-learn-data.pickle", mode="rb") as file:
    data = pickle.load(file)

vectorizer = HashingVectorizer(n_features=2**5)
x_train = vectorizer.transform(data['x_train'])
y_train = data['y_train']
x_test = vectorizer.transform(data['x_test'])
y_test = data['y_test']

clf_bernoulli = BernoulliNB()
clf_bernoulli.fit(x_train, y_train)
y_pred = clf_bernoulli.predict(x_test)
print(f"bernoulli: ({accuracy_score(y_test, y_pred)})")

clf_dtc = DecisionTreeClassifier(max_depth=12)
clf_dtc.fit(x_train, y_train)
y_pred_dtc = clf_dtc.predict(x_test)
print(f"decision tree: ({accuracy_score(y_test, y_pred_dtc)})")

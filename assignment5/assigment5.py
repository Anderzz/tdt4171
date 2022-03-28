from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pickle

#open file
with open(file="scikit-learn-data.pickle", mode="rb") as file:
    data = pickle.load(file)

#preprocessing
vectorizer = HashingVectorizer(n_features=2**20, stop_words='english')
x_train = vectorizer.transform(data['x_train'])
y_train = data['y_train']
x_test = vectorizer.transform(data['x_test'])
y_test = data['y_test']

#define the model
clf_bernoulli = BernoulliNB()
#fit the model to the data
clf_bernoulli.fit(x_train, y_train)
#predict the test data
y_pred = clf_bernoulli.predict(x_test)
print(f"NB: {accuracy_score(y_test, y_pred)}")

#define the model
clf_dtc = DecisionTreeClassifier(criterion='gini', max_depth=18)
#fit the model to the data
clf_dtc.fit(x_train, y_train)
#predict the test data
y_pred_dtc = clf_dtc.predict(x_test)
print(f"decision tree: {accuracy_score(y_test, y_pred_dtc)}")

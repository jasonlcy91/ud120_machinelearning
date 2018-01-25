#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

### this is copied from validate_poi.py!
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import numpy as np

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)
tree = DecisionTreeClassifier()
tree.fit(features_train, labels_train)

pred = tree.predict(features_test)
print metrics.accuracy_score(labels_test,pred)

### your code goes here
# 0 means not POI, 1 means POI. Sum the pred array up to get how many POI has been predicted. Or you can loop through the array to count '1'
print("A. POIs predicted by identifier: ", sum(pred))
print("B. Headcounts in test set: ", len(labels_test))
print("If this identifier predicted everyone as 0, accuracy is (B-A)/B: ", (len(labels_test)-sum(pred))/len(labels_test))

print("Checking for true positives: pred + labels_test")
true_positive = pred + labels_test
print(true_positive)
unique, counts = np.unique(true_positive, return_counts=True)
key = 2
d = dict(zip(unique, counts))
if key in d:
    print("True positives: ", d[key])
else:
    print("True positives: ", 0)

print("Well, accuracy is pretty inaccurate when having imbalanced classes and guessing the more common class.")
print("Let's use precision score, recall score and then compute f1 score.")
print("Precision score: ", metrics.precision_score(labels_test, pred))
print("Recall score: ", metrics.recall_score(labels_test, pred))
print("F1 score: ", metrics.f1_score(labels_test, pred))


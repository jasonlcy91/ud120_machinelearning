#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess # noqa


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

features_train = features_train[:len(features_train)/100]
labels_train = labels_train[:len(labels_train)/100]

#########################################################
### your code goes here ###
from sklearn import svm
from sklearn.metrics import accuracy_score

clf = svm.SVC(kernel="rbf", C=10000)

t0 = time()
clf.fit(features_train, labels_train)
print "Training time is ", round(time()-t0, 3), "s"

t0 = time()
## print(clf.score(features_test, labels_test))
pred = clf.predict(features_test)
print "Testing time is ", round(time()-t0, 3), "s"

elements = [10, 26, 50]
for i in elements:
    if pred[i] == 1:
        print i, ": 1 (Chris)"
    elif pred[i] == 0:
        print i, ": 0 (Sara)"

print "\n"
count = 0
for i in pred:
    if i == 1:
        count += 1

print count
print len(pred)
#########################################################



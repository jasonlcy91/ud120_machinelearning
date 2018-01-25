#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()

### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii] == 0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii] == 0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii] == 1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii] == 1]

#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color="b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color="r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from time import time


def create_clf(classifier, features_train, labels_train):
    switcher = {
        "adaboost": AdaBoostClassifier(DecisionTreeClassifier(min_samples_split=40), n_estimators=200),
        "randomForest": RandomForestClassifier(criterion="entropy", n_estimators=200, warm_start=True, min_samples_split=50),
        "KNearestNeighbors": KNeighborsClassifier(n_neighbors=8, algorithm="auto")
    }

    clf = switcher.get(classifier, "nothing")
    t0 = time()
    clf.fit(features_train, labels_train)
    print "Training time is ", round(time() - t0, 3), "s"
    return clf


def clf_predict(clf, features_test, labels_test):
    t0 = time()
    pred = clf.predict(features_test)
    print "Testing time is ", round(time() - t0, 3), "s"
    accuracy = accuracy_score(labels_test, pred)
    print "Accuracy: ", accuracy
    return pred, accuracy


try:
    clf1 = create_clf("adaboost", features_train, labels_train)
    clf_predict(clf1, features_test, labels_test)

    clf2 = create_clf("randomForest", features_train, labels_train)
    clf_predict(clf2, features_test, labels_test)

    clf3 = create_clf("KNearestNeighbors", features_train, labels_train)
    clf_predict(clf3, features_test, labels_test)

    ## prettyPicture(clf1, features_test, labels_test)
    ## prettyPicture(clf2, features_test, labels_test)
    prettyPicture(clf3, features_test, labels_test)
except NameError:
    pass

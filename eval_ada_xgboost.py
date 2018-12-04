# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 19:29:13 2018

@author: malee
"""
from __future__ import print_function

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


from sklearn.ensemble import AdaBoostClassifier

from xgboost import XGBClassifier
import pandas as pd
from sklearn import model_selection
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.externals import joblib
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot



print(__doc__)

# Loading the Digits dataset
def evaluate_classifier(estimator_name, model, tuned_parameters, X_train, X_test, y_train, y_test):

     scores = ['accuracy']

     for score in scores:
         print("# Tuning hyper-parameters for %s" % score)
         print()

     clf = GridSearchCV(estimator_name(), tuned_parameters, cv=5,
                       scoring='%s' % score)
     clf.fit(X_train, y_train)
     joblib.dump(clf.best_estimator_, str(model)+'.pkl')
     

     print("Best parameters set found on development set:")
     print()
     print(clf.best_params_)
     print()
     print("Grid scores on development set:")
     print()
     means = clf.cv_results_['mean_test_score']
     stds = clf.cv_results_['std_test_score']
     for mean, std, params in zip(means, stds, clf.cv_results_['params']):
         print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
     print()
     print("Detailed classification report for " + str(estimator_name) +":")
     print()
     print("The model is trained on the full development set.")
     print("The scores are computed on the full evaluation set.")
     print()
     y_true, y_pred = y_test, clf.predict(X_test)
     print(classification_report(y_true, y_pred))
     print("Detailed confusion matrix:")
     print(confusion_matrix(y_true, y_pred))
     print("Accuracy Score: \n")
     print(accuracy_score(y_true, y_pred))
      # predict probabilities
     probabilities = clf.predict_proba(X_test)
     # keep probabilities for the positive outcome only
     probabilities = probabilities[:, 1]
     # calculate Area under curve
     auc = roc_auc_score(y_test, probabilities)
     print('Area Under Curve: %.3f' % auc)
     # calculate roc curve
     fpr, tpr, thresholds = roc_curve(y_test, probabilities)
     # plot no skill
     pyplot.plot([0, 1], [0, 1], linestyle='--')
     # plot the roc curve for the model
     pyplot.plot(fpr, tpr, marker='.')
     # show the plot
     pyplot.show()

     print()
     print("*************************************************************************")
    
if __name__ == "__main__":
    df=pd.read_csv('train_data.csv')
    df_test = pd.read_csv('test_data.csv')
df
X_train = df.iloc[:,:2915].values
y_train = df.iloc[:, 2916].values
X_test = df_test.iloc[:,:2915].values
y_test = df_test.iloc[:, 2916].values

estimator_name = [AdaBoostClassifier, XGBClassifier]
model_name = ["AdaBoost Classifier","XGBoost"]
parameters = [
                             [
                                    {#AdaBoost Classifier
                                           'n_estimators':[100,200,350],
                                           'learning_rate':[0.2,0.1,0.09],
                                           'algorithm':['SAMME.R','SAMME'],
                                    }
                               ],
                            
                               [
                                    {#XGBoost
                                           'learning_rate':[0.1, 0.09, 0.5],
                                           'n_estimators':[100, 200, 300],
                                           'booster':['gbtree','gblinear']
                                    }
                               ]
    
                        ]
warnings.simplefilter("ignore")
i=0
for est in estimator_name:
    print("Evaluating Best Parameters and Accuracy for " + model_name[i])
    evaluate_classifier(est, model_name[i], parameters[i], X_train, X_test, y_train, y_test)
    i=i+1
# Note the problem is too easy: the hyperparameter plateau is too flat and the
# output model is the same for precision and recall with ties in quality.
#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import division
"""
Created on Sat Nov 26 09:44:13 2016

@author: densonsmith

I started with the code from a Kaggle 3rd place finisher. The 1st and 2nd place
finishers were only very slightly better on the holdout set and had much more
complex solutions that might not have performed better on a different test set
and certainly would not have generalized to most other problem domains.

The general concept is that if we build multiple different models trained on
different samples of our training data we get multiple predictions that are 
substantially better than chance and that are uncorrelated with each other.

In step 1 we take stratified fold samples of our training data and build multiple 
models (in this case RDF-entropy,RDF-gini ET-entropy,ET-gini and GBT) on each fold. 
We then use the trained models to predict the training sample **not** in the 
training part of this fold. It is super important that you do not use a given
model to predict training data that was used to train that model on that fold.
We also predict all the test data with each model. These predictions are a way 
of transforming the training data and the test data into a different space with 
the predicted probabilities as the transformed information. We take a simple  
average of the predictions of each type of model (eg RDF-gini) and that becomes 
the transformed data for the next step. If we have 5 different models as in this
case our input data for step 2 will have 5 columns and the same number of rows
as the training set and test set respectively.

In step 2 we use a train a logistic regresson on the transformed training data
and use it to predict the transformed test data. We take the predicted probabilities
from the LR as our final answer. 

This method usually results in an improvement over a single highly tuned model
for "hard" problems and not "simple" problems. By hard I mean that the decision
boundary between classes is highly non-linear. Overlapping classes and non-linear
relationships between features contribute to making problems hard.



In this example we are using Matthews correlation coefficient (MCC) and log loss as
performance metrics.

MCC is related to chi squared and is a very good way to measure classification
performance for binary classification problems or a multiclass classification 
problem that has been converted to a series of binary classification problems. 
The main reason is that it works well even with highly imbalanced classes. For 
example if we have a problem where 98% of the rows are class = 0 and 2% are 
class = 1, MCC will usually give you a better picture of performance than F1-score
or precision/recall. The range of MCC is -1 to 1 with -1 being a perfect negative
classifier, 1 a perfect classifier and 0 equal to chance.

While not exactly true, it is convenient to think of MCC as how much better we
did than chance. If you have a very difficult problem and you have an MCC of
30% then our performance is about 30% better than if we had guessed using the class
distribution.

Log Loss is a good way to measure the confidence of your model. In general, even
if we have a very accurate model it is bad if it has high confidence when it is 
wrong. 

For example, if we have a model with the following performance:
    
    
# Model 1
    
y_true = np.array([1,1,1,1,1,0,0,0,0,0])
proba = np.array([0.9,0.9,0.9,0.9,0.1,0.1,0.1,0.1,0.1,0.1])
y_pred = np.array([1,1,1,1,0,0,0,0,0,0])
log_loss = logloss(proba, actual_class, epsilon=1.0e-15)
MCC = matthews_corrcoef(y_true,y_pred)
print('log loss = %.4f' % log_loss)
print('MCC = %.4f' % MCC)

    
    y_true    y_pred        proba
        1       1           0.9
        1       1           0.9
        1       1           0.9
        1       1           0.9
        1       0           0.1  *** way wrong      
        0       0           0.1
        0       0           0.1
        0       0           0.1
        0       0           0.1
        0       0           0.1

log_loss = 0.3251
MCC = 0.8165

Now consider this model:
    
# Model 2
    
y_true = np.array([1,1,1,1,1,0,0,0,0,0])
proba = np.array([0.8,0.8,0.8,0.8,0.4,0.2,0.2,0.2,0.2,0.2])
y_pred = np.array([1,1,1,1,0,0,0,0,0,0])
log_loss = logloss(proba, actual_class, epsilon=1.0e-15)
MCC = matthews_corrcoef(y_true,y_pred)
print('log loss = %.4f' % log_loss)
print('MCC = %.4f' % MCC)

    
    y_true    y_pred        proba
        1       1           0.8
        1       1           0.8
        1       1           0.8
        1       1           0.8
        1       0           0.4  *** slightly wrong         
        0       0           0.2
        0       0           0.2
        0       0           0.2
        0       0           0.2
        0       0           0.2

log_loss = 0.2925
MCC = 0.8165

Model 2 is better because although it is less confident when it is right it is
much less confident when it is wrong.
        

Kaggle competition: Predicting a Biological Response.

Blending {RandomForests, ExtraTrees, GradientBoosting} + stretching to
[0,1]. The blending scheme is related to the idea Jose H. Solorzano
presented here:
http://www.kaggle.com/c/bioresponse/forums/t/1889/question-about-the-process-of-ensemble-learning/
'''You can try this: In one of the 5 folds, train the models, then use
the results of the models as 'variables' in logistic regression over
the validation data of that fold'''. Or at least this is the
implementation of my understanding of that idea :-)

The predictions are saved in test.csv. The code below created my best
submission to the competition:
- public score (25%): 0.43464
- private score (75%): 0.37751
- final rank on the private leaderboard: 17th over 711 teams :-)

Note: if you increase the number of estimators of the classifiers,
e.g. n_estimators=1000, you get a better score/rank on the private
test set.

Copyright 2012, Emanuele Olivetti.
BSD license, 3 clauses.





"""


from __future__ import division


import sklearn

import numpy as np

sklearn_version = sklearn.__version__
print('The scikit-learn version is {}.'.format(sklearn.__version__))

sklearn_version = sklearn_version.split('.')
main_sklearn_verison = int(sklearn_version[1])

current_scikit_verison_flag = True

if main_sklearn_verison < 18:
    print('Your version of scikit learn is less than version 18.')
    print('Denson will stop supporting versions less than 18 in March 2017.')
    current_scikit_verison_flag = False





import pandas as pd

if current_scikit_verison_flag:
    from sklearn.model_selection import StratifiedKFold,StratifiedShuffleSplit
else:
    from sklearn.cross_validation import StratifiedKFold, StratifiedShuffleSplit
    
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import matthews_corrcoef


from sklearn.datasets import  make_classification


def logloss(attempt, actual, epsilon=1.0e-15):
    """Logloss
    """
    attempt = np.clip(attempt, epsilon, 1.0-epsilon)
    return - np.mean(actual * np.log(attempt) +
                     (1.0 - actual) * np.log(1.0 - attempt))
    
    
    


if __name__ == '__main__':

    np.random.seed(0)  # seed to shuffle the train set
    
                  
                  

    n_folds = 10
    verbose = True
    shuffle = False
    
    

    
    

    
    # In a real problem remove the code between these comments and load your data
    '''
    Generate a classification problem. 
    This is a difficult problem because:
    
        1) There are only 10 relevant features and 440 noise features
        2) There is a great deal of overlap between classes
        3) There are 3 clusters per class
        4) The relationships between featues is non-linear
        5) We have a fairly small sample size for the complexity of the problem.
        
    Noise features will tend to confuse any machine learning model. In this case
    we have far more noise features that contain signal.
    
    The class_sep parameter controls the distance between the center of each
    cluster. There is gausian noise within each cluster. When the class_sep < 1 
    there is a good deal of overlap between classes and the blended model will 
    be best. If the class_sep > 1 then a single model will often beat the blended 
    model. If the class_sep > 2 then a simple model like logistic regression will 
    usually be best.
    
    Having multiple clusters per class means that there is certainly no linear
    boundary between classes. Also, this compounds to problem of overlap between
    classes.
    
    The SKLEARN make_classification is designed to generate non-linear relationships
    between features based on the algorithm described in :
    Guyon, Isabelle. "Design of experiments of the NIPS 2003 variable selection benchmark." 
    
    Empirically, as the sample size increases the classification performance
    will also increase but if the blended model wins it will continue to win.
    
    Notice that we are not doing any tuning. It might be possible to tune
    a GLM or other model to outperform the blended model on a particular sample
    of data. However, that highly tuned model will likely not generalize as well
    for new data. In a real problem our available sample for training and testing
    a model is nearly always biased in some unknown way. A blended model is 
    much more robust to this.
        
    '''

    sample_size = 10000
    n_features = 450
    n_informative = 10 
    num_noise_feats = n_features - n_informative
    class_sep = 0.5
    mislabel = 0
    
    # This will create mislabled/noisy y labels.
    flip_y = float(mislabel)/100.0
    # Generate the problem
    X_gen, y_gen = make_classification(n_samples=sample_size , 
                               n_features=n_features , 
                               n_redundant=0, 
                               n_informative=n_informative,
                               random_state=42, 
                               n_clusters_per_class=3, 
                               class_sep = class_sep, 
                               flip_y = flip_y, 
                               shuffle = True)
    
    
    # Use SSS to create training and test sets
    
    if current_scikit_verison_flag:
        sss = StratifiedShuffleSplit(n_splits=1,
                                     test_size=0.2,
                                     random_state=42)
        
        for train_index,test_index in sss.split(X_gen,y_gen):
            X_train = X_gen[train_index] 
            y_train = y_gen[train_index]
    
            X_test = X_gen[test_index]
            y_test = y_gen[test_index]
    
    else:
        sss = StratifiedShuffleSplit(y_gen, 1, test_size=0.2, random_state=42)
    
        for train_index,test_index in sss:
            X_train = X_gen[train_index] 
            y_train = y_gen[train_index]
    
            X_test = X_gen[test_index]
            y_test = y_gen[test_index]
        

    


    y_holdout_true = y_test[:]

    X_holdout = X_test[:,:]

    # In a real problem remove the code between these comments and load your data

    
    
    # We run the classifiers with different parameters to make the predictions
    # less correlated.
    clfs = [RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
            RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
            ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
            ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
            GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=50)]

    print "Creating train and test sets for blending."
    
    # These arrays will hold the blended predictions
    dataset_blend_train = np.zeros((X_train.shape[0], len(clfs)))
    dataset_blend_test = np.zeros((X_holdout.shape[0], len(clfs)))
    
    # Use SSS to create training and test sets
    
    if current_scikit_verison_flag:
        # For each fold train on 80% and test on 20%. If you have a bunch of data
        # you might want to make it a 50/50 split.
        sss = StratifiedShuffleSplit(n_splits=n_folds,
                                     test_size = 0.2)
    
    
        # We will resuse the same train-test splits for each of the models.
        splits = list(sss.split(X_train,y_train))
    
    else:
        sss = StratifiedShuffleSplit(y_train, n_folds, test_size=0.2)
        
        splits = list(sss)
    
    for jdx, clf in enumerate(clfs):
        print jdx, clf
        
        # dataset__blend_test_j is for this fold of this model (RDF-gini, fold 1 etc)
        dataset_blend_test_j = np.zeros((X_holdout.shape[0], len(splits)))
        for idx, (train, test) in enumerate(splits):
            print "Fold", idx
            
            # Split the training data into train-test sets for this fold
            X_fold_train = X_train[train]
            y_fold_train = y_train[train]
            X_fold_test = X_train[test]
            y_fold_test = y_train[test]
            
            # Fit this model on this fold of data
            clf.fit(X_fold_train, y_fold_train)
            
            # Predict this test fold
            y_fold_pred = clf.predict_proba(X_fold_test)[:, 1]
            
            '''
            This is where things get slightly confusing. We are using part of the
            training data to predict the rest of the training data. We store the
            predictions as the transformed training data. A given row of the 
            training data is likely to be predicted more than once and will wind
            up only with the last prediction. There is no absolute guarantee that 
            every row of the training data will be predicted but it is highly
            likely in 10 folds.
            '''
            # Store the predictions as transformed training data
            # jdx is the index of this model (RDF-gini)
            dataset_blend_train[test, jdx] = y_fold_pred 
                               
            '''
            Now we use this model and use it to predict the holdout test data.
            We store the predictions of each fold of this model and take the mean
            at the end to create the transformation for this model.
            '''
            dataset_blend_test_j[:, idx] = clf.predict_proba(X_holdout)[:, 1]
        
        # Take the mean prediction of each fold and use it as the transformed
        # test data.
        dataset_blend_test[:, jdx] = dataset_blend_test_j.mean(1)

    '''
    Now we train a logistic regression (or some other simple model) on the transformed
    training data and use it to predict the transformed test data.
    '''
    print
    print "Blending."
    clf = LogisticRegression()
    clf.fit(dataset_blend_train, y_train)
    y_holdout = clf.predict_proba(dataset_blend_test)[:, 1]

    '''
    It is possible that the predictions from the logistic regression on the 
    transformed data will be skewed towards 0 or 1. This will stretch the predictions
    back to a range of 0-1
    '''
    print "Linear stretch of predictions to [0,1]"
    y_holdout = (y_holdout - y_holdout.min()) / (y_holdout.max() - y_holdout.min())
    
    y_pred = np.zeros(len(y_holdout))
    
    # Convert the probabilities to  integer predictions of class 0 or class 1
    class_one_rows = np.where(y_holdout >= 0.5)[0]

    y_pred[class_one_rows] = 1
    
    # Compute some performance metrics
    MCC = matthews_corrcoef(y_holdout_true,y_pred)
    print('Blended MCC = %.4f' % MCC)
    
    log_loss = logloss(y_holdout, y_holdout_true, epsilon=1.0e-15)
    print('Blended logloss = %.4f' % log_loss)    
    
    
    # Train single models for comparison
    clf = RandomForestClassifier(n_estimators=100, 
                                 n_jobs=-1, 
                                 criterion='entropy')
    clf.fit(X_train,y_train)
    y_proba = clf.predict_proba(X_test)[:, 1]
    
    y_pred = np.zeros(len(y_holdout_true))
    
    class_1_rows = np.where(y_proba >= 0.5)[0]

    y_pred[class_1_rows] = 1
    
    
    MCC = matthews_corrcoef(y_holdout_true,y_pred)
    print('Single RDF MCC = %.4f' % MCC)
    
    llog_loss = logloss(y_proba, y_holdout_true, epsilon=1.0e-15)
    print('Single RDF logloss = %.4f' % log_loss) 
    
    
    clf = ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='gini')
    clf.fit(X_train,y_train)
    y_proba = clf.predict_proba(X_test)[:, 1]
    
    y_pred = np.zeros(len(y_holdout_true))
    
    class_1_rows = np.where(y_proba >= 0.5)[0]

    y_pred[class_1_rows] = 1
    
    
    MCC = matthews_corrcoef(y_holdout_true,y_pred)
    print('Single ET MCC = %.4f' % MCC)

    log_loss = logloss(y_proba, y_holdout_true, epsilon=1.0e-15)
    print('Single ET logloss = %.4f' % log_loss)     
    
    clf = GradientBoostingClassifier(learning_rate=0.05, 
                                     subsample=0.5, 
                                     max_depth=6, 
                                     n_estimators=50)
    clf.fit(X_train,y_train)
    y_proba = clf.predict_proba(X_test)[:, 1]
    
    y_pred = np.zeros(len(y_holdout_true))
    
    class_1_rows = np.where(y_proba >= 0.5)[0]

    y_pred[class_1_rows] = 1
    
    
    MCC = matthews_corrcoef(y_holdout_true,y_pred)
    print('Single GBT MCC = %.4f' % MCC) 
    
    log_loss = logloss(y_proba, y_holdout_true, epsilon=1.0e-15)
    print('Single GBT logloss = %.4f' % log_loss) 
    
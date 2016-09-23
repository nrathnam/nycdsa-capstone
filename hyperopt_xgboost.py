# -*- coding: utf-8 -*-
"""
Created on Sat Sep 10 13:22:30 2016

@author: trichna
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Sep 06 14:51:05 2016

@author: trichna
"""

from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.metrics import log_loss, auc
from sklearn import preprocessing
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn import ensemble
import numpy as np
import pandas as pd
import sys, os
#import xgboost as xgb
y_column = 'SeriousDlqin2yrs'
replace_val = np.nan 
replace_by = -999

fl = open('C:/Users/trichna/Documents/NYCDSA/Capstone/data/hyperout_gbm.txt','w')

def load_train():
    
    data, Xcol, ycol = process_data('C:/Users/trichna/Documents/NYCDSA/Capstone/data/train.csv',True,y_column, replace_val, replace_by )    
    train = Xcol
    labels = ycol
    print "train DF datatype = ", train.dtype
    
    train = np.delete(train,0,axis=1)
    
    return train, labels.astype('int32')


def load_test():
    test = pd.read_csv('C:/Users/trichna/Documents/NYCDSA/Capstone/data/test.csv')
    test = test.drop('id', axis=1)
    test = test.drop('SeriousDlqin2yrs', axis=1)
    return test.values


def write_submission(preds, output):
    sample = pd.read_csv('C:/Users/trichna/Documents/NYCDSA/Capstone/data/sampleEntry.csv')
    preds = pd.DataFrame(
        preds, index=sample.id.values, columns=sample.columns[1:])
    preds.to_csv(output, index_label='id')

def process_data(filename, training_flag, y_column, replace_val, replace_by):
    """
    Reads in training data and prepares numpy arrays.
    """
    data = pd.read_csv(filename, sep=',', index_col=0)

    X = data.drop([y_column], axis=1).values

    if training_flag:
        y = data[y_column].values


    if replace_val is not None:
        print '\n'
        print 'replacing...'
        if np.isnan(replace_val):
            X[np.where(np.isnan(X))] = replace_by
        else:
            X[X == replace_val] = replace_by
        print 'Total entries of ' + str(replace_val) + ': ' + str(np.sum(X==replace_val))
        print 'Total entries of' + str(replace_by) + ': ' + str(np.sum(X==replace_by))
        print '\n'
             

    

    if training_flag:
        # create a standardization transform
        return data, X, y #, scaler, pca
    else: # retdata, urn test data
        return data, X



def score(params):
    print >> fl, "Training with params : "
    print >> fl, params
    num_round = int(params['n_estimators'])
    del params['n_estimators']
    model = ensemble.GradientBoostingClassifier(params)
    model.fit(X_train,y_train)
    predictions = model.predict_proba(X_test)[: , 1]

    score = metrics.roc_auc_score(y_test, predictions)
    
    
    print >> fl, "\tScore {0}\n\n".format(score)
    return {'loss': -score}


def optimize(trials):
    space = {
             'n_estimators' : hp.quniform('n_estimators', 100, 1000, 1),
             'max_depth' : hp.choice('max_depth', np.arange(1, 14, dtype=int)),
             'min_samples_split' : hp.quniform('min_samples_split', 100, 700, 25),
             'min_samples_leaf' : hp.quniform('min_samples_leaf', 100, 700, 25),
             'max_features' : hp.quniform('max_features', 0.01, 1, 0.01),
             'learning_rate' : hp.quniform('learning_rate', 0.025, 0.8, 0.025),
             'subsample' : hp.quniform('subsample', 0.5, 1, 0.05),
             'loss' : 'deviance',
             'verbose' : 1
             } 
            
             


    best = fmin(score, space, algo=tpe.suggest, trials=trials, max_evals=250)

    #print best
    print >> fl, 'Best Parameter combination:'
    print >> fl, best
    fl.close()



   
X, y = load_train()
print >> fl, "Splitting data into train and valid ...\n\n"
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234)

#Trials object where the history of search will be stored
trials = Trials()

optimize(trials)
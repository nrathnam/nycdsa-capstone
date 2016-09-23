# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 07:54:53 2016

@author: trichna
"""

# -*- coding: utf-8 -*-
"""
@author: trichna
"""

import os, math, time, pickle, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn import decomposition
from sklearn import ensemble
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn import preprocessing

import os
mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-6.2.0-posix-seh-rt_v5-rev1\\mingw64\\bin'
os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc



def load(alg, filename):
    """
    Load a previously training model from disk.
    """
    if alg == 'xgboost':
        model = xgb.Booster({'nthread': 16}, model_file=filename)
    else:
        model_file = open(filename, 'rb')
        model = pickle.load(model_file)
        model_file.close()

    return model


def save(alg, model, filename):
    """
    Persist a trained model to disk.
    """
    if alg == 'xgboost':
        model.save_model(filename)
    else:
        model_file = open(filename, 'wb')
        pickle.dump(model, model_file)
        model_file.close()


def process_training_data(filename, features, impute, standardize, whiten):
    """
    Reads in training data and prepares numpy arrays.
    """
    #training_data1 = pd.read_csv(filename, sep=',')
    training_data = pd.read_csv(filename, sep=',')


    X = training_data.iloc[:, 0:features].values
    y = training_data.iloc[:, features].values

    if impute == 'mean':
        imp = preprocessing.Imputer(missing_values=-999, strategy="mean")
        X = imp.fit_transform(X)
    elif impute == "median":
        imp = preprocessing.Imputer(missing_values=-999, strategy="median")
        X = imp.fit_transform(X)
    elif impute == "most_frequent":
        imp = preprocessing.Imputer(missing_values=-999, strategy="most_frequent")
        X = imp.fit_transform(X)

    elif impute == 'zeros':
        X[X == -999] = 0

    # create a standardization transform
    scaler = None
    if standardize:
        scaler = preprocessing.StandardScaler()
        scaler.fit(X)

    # create a PCA transform
    pca = None
    if whiten:
        pca = decomposition.PCA(whiten=True)
        pca.fit(X)

    #return training_data, X, y, w, scaler, pca
    return training_data, X, y, scaler, pca


def visualize(training_data, X, y, scaler, pca, features):
    """
    Computes statistics describing the data and creates some visualizations
    that attempt to highlight the underlying structure.
    Note: Use '%matplotlib inline' and '%matplotlib qt' at the IPython console
    to switch between display modes.
    """

    # feature histograms
    fig1, ax1 = plt.subplots(4, 4, figsize=(20, 10))
    for i in range(16):
        ax1[i % 4, i / 4].hist(X[:, i])
        ax1[i % 4, i / 4].set_title(training_data.columns[i + 1])
        ax1[i % 4, i / 4].set_xlim((min(X[:, i]), max(X[:, i])))
    fig1.tight_layout()

    fig2, ax2 = plt.subplots(4, 4, figsize=(20, 10))
    for i in range(16, features):
        ax2[i % 4, (i - 16) / 4].hist(X[:, i])
        ax2[i % 4, (i - 16) / 4].set_title(training_data.columns[i + 1])
        ax2[i % 4, (i - 16) / 4].set_xlim((min(X[:, i]), max(X[:, i])))
    fig2.tight_layout()

    # covariance matrix
    if scaler is not None:
        X = scaler.transform(X)

    cov = np.cov(X, rowvar=0)

    fig3, ax3 = plt.subplots(figsize=(16, 10))
    p = ax3.pcolor(cov)
    fig3.colorbar(p, ax=ax3)
    ax3.set_title('Feature Covariance Matrix')

    # pca plots
    if pca is not None:
        X = pca.transform(X)

        fig4, ax4 = plt.subplots(figsize=(16, 10))
        ax4.scatter(X[:, 0], X[:, 1], c=y)
        ax4.set_title('First & Second Principal Components')

        fig5, ax5 = plt.subplots(figsize=(16, 10))
        ax5.scatter(X[:, 1], X[:, 2], c=y)
        ax5.set_title('Second & Third Principal Components')


#def train(X, y, w, alg, scaler, pca):
def train(X, y, alg, scaler, pca):
    """
    Trains a new model using the training data.
    """
    if scaler is not None:
        X = scaler.transform(X)

    if pca is not None:
        X = pca.transform(X)

    if alg == 'xgboost':
        # use a separate process for the xgboost library
        #return train_xgb(X, y, w, scaler, pca)
        return train_xgb(X, y, scaler, pca)

    t0 = time.time()

    if alg == 'bayes':
        model = naive_bayes.GaussianNB()
    elif alg == 'logistic':
        model = linear_model.LogisticRegression()
    elif alg == 'svm':
        model = svm.SVC(probability = True)

    elif alg == 'boost':
        model = ensemble.GradientBoostingClassifier(n_estimators=200, max_depth=7,
            min_samples_split=200, min_samples_leaf=200, max_features=0.2)
        #model = ensemble.GradientBoostingClassifier(n_estimators=197, max_depth=5,
        #    min_samples_split=319, min_samples_leaf=89, max_features=0.2)
    elif alg == 'forest':
        #model = ensemble.RandomForestClassifier(n_estimators=100, max_depth=5,
        #    min_samples_split=200, min_samples_leaf=200, max_features=10)
        model = ensemble.RandomForestClassifier(n_estimators=161,  criterion='gini',  min_samples_split=223,
            min_samples_leaf=9, max_features=1,  max_depth=14)
    elif alg == "adaboost":
        model = ensemble.AdaBoostClassifier(n_estimators=100, learning_rate=0.1)
    else:
        print 'No model defined for ' + alg
        exit()

    model.fit(X, y)

    t1 = time.time()
    print 'Model trained in {0:3f} s.'.format(t1 - t0)

    return model


#def train_xgb(X, y, w, scaler, pca):
def train_xgb(X, y, scaler, pca):
    """
    Trains a boosted trees model using the XGBoost library.
    """
    t0 = time.time()

    #xgmat = xgb.DMatrix(X, label=y, missing=-999.0, weight=w)
    xgmat = xgb.DMatrix(X, label=y, missing=-999.0)


    param = {}
    #param['objective'] = 'binary:logitraw'
    param['n_estimators'] = 101
    param['objective'] = 'binary:logistic'
    param['eta'] =   0.1    #0.08
    param['colsample_bytree'] = 0.95
    param['min_child_weight'] = 4
    param['gamma'] = 0.85
    param['max_depth'] =   4    #3
    param['subsample'] =   0.95    #0.8
    param['eval_metric'] = 'auc'
    param['silent'] = 1  
    #rounds = 500 
    

    
    plst = list(param.items())
    watchlist = []

    #model = xgb.train(plst, xgmat, 128, watchlist)
    model = xgb.train(plst, xgmat, 128, watchlist)

    t1 = time.time()
    print 'Model trained in {0:3f} s.'.format(t1 - t0)

    return model


def predict(X, model, alg, threshold, scaler, pca):
    """
    Predicts the probability of a positive outcome and converts the
    probability to a binary prediction based on the cutoff percentage.
    """
    if scaler is not None:
        X = scaler.transform(X)

    if pca is not None:
        X = pca.transform(X)

    if alg == 'xgboost':
        xgmat = xgb.DMatrix(X, missing=-999.0)
 
        y_prob = model.predict(xgmat)
    else:
        y_prob = model.predict_proba(X)[:, 1]

    cutoff = np.percentile(y_prob, threshold)
    y_est = y_prob > cutoff
    print "after checking CUTOFF..."
    print 'y prob = ', y_prob, len(y_prob)
    print 'y_estimated =', y_est, len(y_est)
    return y_prob, y_est



def score(y, y_est):
    return calculate_auc(y, y_est)
    
def a_score(y, y_pred_prob):
    """
    updated score function to plot ROC curve
    Create weighted signal and background sets and calculate the AMS.
    """
    #print 'within score w shape=', w.shape
    print 'within score y shape=', y.shape
    print 'within score y estimated shape=',  y_pred_prob.shape

    fpr, tpr, _ = roc_curve(y, y_pred_prob)
    aucscore = auc(fpr,tpr)
    print 'AUC value =', aucscore
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % aucscore)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    
    return aucscore



def cross_validate(X, y, alg, scaler, pca, threshold):
    """
    Perform cross-validation on the training set and compute the AMS scores.
    """
    import warnings
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

    
    scores = [0, 0, 0,0,0]
    
    folds = cross_validation.StratifiedKFold(y, n_folds=5)
    i = 0

    for i_train, i_val in folds:
        # create the training and validation sets
        X_train, X_val = X[i_train], X[i_val]
        y_train, y_val = y[i_train], y[i_val]
        #w_train, w_val = w[i_train], w[i_val]

     
        model = train(X_train, y_train, alg, scaler, pca)

        # predict and score performance on the validation set
        y_val_prob, y_val_est = predict(X_val, model, alg, threshold, scaler, pca)

        scores[i] = score(y_val, y_val_prob)
        i += 1

    return np.mean(scores)
    #return model, np.mean(scores)

def write_submission(y_test_prob, output):
    sample = pd.read_csv('C:/Users/trichna/Documents/NYCDSA/Capstone/data/sampleEntry.csv')
    preds = pd.DataFrame(
        y_test_prob, index=sample.Id.values, columns=sample.columns[1:])
    preds.to_csv(output, index_label='Id')
    

def process_test_data(filename, features, impute):
    """
    Reads in test data and prepares numpy arrays.
    """
    
    test_data1 = pd.read_csv(filename, sep=',')
    test_data = test_data1.sample(frac = 1)
    X_test = test_data.iloc[:, 0:features].values

 
    if impute == 'mean':
        imp = preprocessing.Imputer(missing_values=-999, strategy="mean")
        X = imp.fit_transform(X)
    elif impute == "median":
        imp = preprocessing.Imputer(missing_values=-999, strategy="median")
        X = imp.fit_transform(X)
    elif impute == "most_frequent":
        imp = preprocessing.Imputer(missing_values=-999, strategy="most_frequent")
        X = imp.fit_transform(X)

    elif impute == 'zeros':
        X[X == -999] = 0

    return test_data, X_test

def process_data(filename, training_flag, features, impute, standardize, whiten, y_column, replace_val, replace_by):
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
             

 
    if impute == 'mean':
        imp = preprocessing.Imputer(missing_values = 'NaN', strategy="mean")
        X = imp.fit_transform(X)
    elif impute == "median":
        imp = preprocessing.Imputer(missing_values = 'NaN', strategy="median")
        X = imp.fit_transform(X)
    elif impute == "most_frequent":
        imp = preprocessing.Imputer(missing_values = 'NaN', strategy="most_frequent")
        X = imp.fit_transform(X)
    elif impute == "knn":
        X = KNN(k = math.sqrt(len(X))).complete(X)

    elif impute == 'zeros':
        print '\n'
        print 'Total number of zeros: ' + str(np.sum(X==0))
        print 'Imputing...'
        X[np.where(np.isnan(X))] = 0
        print 'Total number of zeros: ' + str(np.sum(X==0))
        print '\n'
    elif impute == 'none':
        pass
    else:
        print 'Error: Imputation method not found.'
        quit()

    if training_flag:
        # create a standardization transform
        scaler = None
        if standardize:
            scaler = preprocessing.StandardScaler()
            scaler.fit(X)

        # create a PCA transform
        pca = None
        if whiten:
            pca = decomposition.PCA(whiten=True)
            pca.fit(X)
    
        return data, X, y, scaler, pca
    else: # return test data
        return data, X




def create_submission(test_data, y_test_prob, y_test_est, submit_file):
    """
    Create a new data frame with the submission data.
    """
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

    submit = pd.DataFrame({'Id': range(1, len(y_test_prob)+1), 'Probability': y_test_prob})

    # finally create the submission file
    submit.to_csv(submit_file, sep=',', index=False, index_label=False)

def calculate_auc(y, y_pred_prob, plot_flag = False):
    """
    updated score function to plot ROC curve
    Create weighted signal and background sets and calculate the AMS.
    """
    #print 'within score w shape=', w.shape
    print 'within score y shape=', y.shape
    print 'within score y estimated shape=',  y_pred_prob.shape  #y_est.shape
    fpr, tpr, thresholds = roc_curve(y, y_pred_prob, pos_label = 1)
    aucscore = auc(fpr, tpr)
    print 'AUC value =', aucscore
    print 'Thresholds = ', thresholds

    #visualize roc 
    if plot_flag:
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % aucscore)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()

    return aucscore

def predict_ens(X, model, alg, threshold, scaler, pca):
    """
    Predicts the probability of a positive outcome and converts the
    probability to a binary prediction based on the cutoff percentage.
    """
    if scaler is not None:
        X = scaler.transform(X)

    if pca is not None:
        X = pca.transform(X)

    if alg == 'xgboost':
        xgmat = xgb.DMatrix(X, missing=-999.0)
        
        y_prob = model.predict(xgmat)
        
    else:
        y_prob = model.predict_proba(X)[:, 1]

   

    return y_prob

def cross_validate_ens(X, y, w, j, algo, scaler, pca, threshold, dataset_blend_train):
    """
    Perform cross-validation on the training set and compute the AMS scores.
    """
    import warnings
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

    scores = [0, 0, 0]
    
    folds = cross_validation.StratifiedKFold(y, n_folds=5)
    i = 0

    for i_train, i_val in folds:
        # create the training and validation sets
        X_train, X_val = X[i_train], X[i_val]
        y_train, y_val = y[i_train], y[i_val]
        w_train, w_val = w[i_train], w[i_val]

        # normalize the weights
        w_train[y_train == 1] *= (sum(w[y == 1]) / sum(w[y_train == 1]))
        w_train[y_train == 0] *= (sum(w[y == 0]) / sum(w_train[y_train == 0]))
        w_val[y_val == 1] *= (sum(w[y == 1]) / sum(w_val[y_val == 1]))
        w_val[y_val == 0] *= (sum(w[y == 0]) / sum(w_val[y_val == 0]))

        # train the model
        model = train(X_train, y_train, w_train, algo, scaler, pca)

        # predict and score performance on the validation set
        y_val_prob = predict_ens(X_val, model, algo, threshold, scaler, pca)
        dataset_blend_train[i_val,j] = y_val_prob
     

    return np.mean(scores)



def main():
    # perform some initialization
    features = 11  #30
    threshold = 50
    alg = 'xgboost'  # bayes, logistic, svm, boost, xgboost
    stack_ensemble =  True
    impute = 'none'  
    standardize = False
    whiten = False

    code_dir = 'C:/Users/trichna/Documents/Python Scripts/ConsumerDebt/'
    data_dir = 'C:/Users/trichna/Documents/NYCDSA/Capstone/data/'
    training_file =  'cs-training.csv'  #'cs-training-2.csv'  
    test_file =   'cs-test.csv'   #'cs-test-2.csv' 
    submit_file = 'submission.csv'
    model_file = 'model1.pkl'
    alg_list =    ['forest','xgboost']   #['boost','xgboost']
    load_training_data =  False
    load_model = False
    train_model =  False
    save_model = False
    create_visualizations = False
    create_submission_file = False
    y_column = 'SeriousDlqin2yrs'
    replace_val = np.nan # -999 for Higgs Boson 
    replace_by = -999
    plot_flag = False
    os.chdir(code_dir)

    print 'Starting process...'
    print 'alg={0}, impute={1}, standardize={2}, whiten={3}, threshold={4}'.format(
        alg, impute, standardize, whiten, threshold)
   
    if load_training_data:
        print 'Reading in training data...'

        training_data, X, y, scaler, pca = process_data(
            data_dir + training_file, True, features, impute, standardize, whiten, y_column, replace_val, replace_by)  


    if create_visualizations:
        print 'Creating visualizations...'
        visualize(training_data, X, y, scaler, pca, features)

    if load_model:
        print 'Loading model from disk...'
        model = load(alg, data_dir + model_file)

    if train_model:
        print 'Training model on full data set...'
        
        model = train(X, y, alg, scaler, pca)

        print 'Calculating predictions...'
        y_prob, y_est = predict(X, model, alg, threshold, scaler, pca)

        print 'Calculating AUC...'
        auc_val = score(y, y_prob)
        print 'AUC =', auc_val

        print 'Performing cross-validation...'
        
        val = cross_validate(X, y, alg, scaler, pca, threshold)
        
        print'Cross-validation AUC =', val

    if save_model:
        print 'Saving model to disk...'
        save(alg, model, data_dir + model_file)

    if create_submission_file:
        print 'Reading in test data...'
        
        test_data, X_test = process_data(
            data_dir + test_file, False, features, impute, standardize, whiten, y_column, replace_val, replace_by)


        print 'Predicting test data...'
        y_test_prob, y_test_est = predict(X_test, model, alg, threshold, scaler, pca)
        

        print 'Creating submission file...'
        create_submission(test_data, y_test_prob, y_test_est, data_dir + submit_file)
        

    
    
    if stack_ensemble:
        training_data, X, y, scaler, pca = process_data(
            data_dir + training_file, True, features, impute, standardize, whiten, y_column, replace_val, replace_by)  
        
        
        test_data, X_test = process_data(
            data_dir + test_file, False, features, impute, standardize, whiten, y_column, replace_val, replace_by)

        dataset_blend_train = np.zeros((X.shape[0], len(alg_list))) 
        dataset_blend_test = np.zeros((X_test.shape[0], len(alg_list)))

        print 'dataset_blend_train TRAIN SHAPE =',dataset_blend_train.shape, X.shape, y.shape
        print 'dataset_blend_test TEST SHAPE = ', dataset_blend_test.shape
        for j,algo in enumerate(alg_list):
            

            import warnings
            warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
            print 'algorithm = ', algo
           
            folds = cross_validation.StratifiedKFold(y, n_folds=5)
            i = 0
            dataset_blend_test_j = np.zeros((X_test.shape[0], len(folds)))
            print 'dataset_blend_test_jj shape  --', dataset_blend_test_j.shape
            for i, (i_train, i_val) in enumerate(folds):
                # create the training and validation sets
                X_train, X_val = X[i_train], X[i_val]
                y_train, y_val = y[i_train], y[i_val]
        
               
                model = train(X_train, y_train,  algo, scaler, pca)
        
                # predict and score performance on the validation set
                y_val_prob = predict_ens(X_val, model, algo, threshold, scaler, pca)
                dataset_blend_train[i_val,j] = y_val_prob
                y_test_prob = predict_ens(X_test, model, algo, threshold, scaler, pca)
                dataset_blend_test_j[:, i] = y_test_prob

            dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)
            
        bclf = linear_model.LogisticRegression() 
        model_ens = bclf.fit(dataset_blend_train, y) 
        y_val_prob_ens, y_val_est_ens = predict(dataset_blend_train, model_ens, 'logistic', threshold, scaler, pca)

        auc_val = score(y, y_val_prob_ens)
        print 'Cross-validation Ensemble AUC =', auc_val
        print 'Predicting test data...'
        y_test_prob_ens, y_test_est_ens = predict(dataset_blend_test, model_ens, 'logistic', threshold, scaler, pca)

        print 'Creating submission file...'
        create_submission(test_data, y_test_prob_ens, y_test_est_ens, data_dir + submit_file)        
        

    print 'Process complete.'



if __name__ == "__main__":
    main()
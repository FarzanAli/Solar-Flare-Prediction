import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn import svm as support_vector_machine
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


class my_svm():
    # __init__() function should initialize all your variables
    def __init__(self, X, Y, feature_set):
        self.target = Y
        self.input = X

        res = self.feature_creation(feature_set)
        self.input = res

    # preprocess() function:
    #  1) normalizes the data, 
    #  2) removes missing values
    #  3) assign labels to target 
    def preprocess(self):
        scalar = StandardScaler().fit(self.input)

        X_ = scalar.transform(self.input)
        #Labels negative solar events as -1.0 (instead of None) and positive solar events as 1.0
        Y_ = [ -1.0 if c is None else 1.0 for harp_number, peak_flare_time, c in self.target ]

        return np.array(X_), np.array(Y_)
    
    # feature_creation() function takes as input the features set label (e.g. FS-I, FS-II, FS-III, FS-IV)
    # and creates 2 D array of corresponding features 
    # for both positive and negative observations.
    # this array will be input to the svm model
    # For instance, if the input is FS-I, the output is a 2-d array with features corresponding to 
    # FS-I for both negative and positive class observations

    #Concatenates feature set values given labels
    def feature_creation(self, fs_value):
        X_ = []
        for fs in fs_value:
            X_.append(self.input[fs])
        X_ = np.concatenate(X_, axis=1)
        return X_
    
    # cross_validation() function splits the data into train and test splits,
    # Use k-fold with k=10
    # the svm is trained on training set and tested on test set
    # the output is the average accuracy across all train test splits.
    def cross_validation(self, svc, X, Y, k):
        # call training function
        # call tss function
        kf = KFold(n_splits=k)

        tss = []
        #Performs k-fold cross validation
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]

            svc.fit(X_train, Y_train)
            cm, score = self.tss(svc, X_test, Y_test)
            tss.append(float(score))
        return kf, tss

    
    #training() function trains a SVM classification model on input features and corresponding target
    def training(self, X, Y):
        svc = support_vector_machine.SVC().fit(X, Y)
        return svc
    
    # tss() function computes the accuracy of predicted outputs (i.e target prediction on test set)
    # using the TSS measure given in the document
    #Uses sklearn metrics to compute tss
    def tss(self, svc, X, Y):
        svc = self.training(X, Y)
        Y_predicted = svc.predict(X)
        cm = confusion_matrix(Y, Y_predicted)
        tn, fp, fn, tp = cm.ravel()
        tss = (tp/(tp + fn)) - (fp/(fp + tn))
        return cm, tss
    

# feature_experiment() function executes experiments with all 4 feature sets.
# svm is trained (and tested) on 2010 dataset with all 4 feature set combinations
# the output of this function is : 
#  1) TSS average scores (mean std) for k-fold validation printed out on console.
#  2) Confusion matrix for TP, FP, TN, FN. See assignment document 
#  3) A chart showing TSS scores for all folds of CV. 
#     This means that for each fold, compute the TSS score on test set for that fold, and plot it.
#     The number of folds will decide the number of points on this chart (i.e 10)
#
# Above 3 charts are produced for all feature combinations
#  4) The function prints the best performing feature set combination
def feature_experiment():
    X, Y = load_data('data-2010-15')
    allSets = np.array(['FS-I', 'FS-II', 'FS-III', 'FS-IV'])
    highest_tss = float('-inf')
    best_feature_set = []
    print("Feature Set: TSS Scores")
    k = 10

    combinations =  [
        [0],
        [1],
        [2],
        [3],
        [0, 1],
        [0, 2],
        [0, 3],
        [1, 2],
        [1, 3],
        [2, 3],
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3],
        [0, 1, 2, 3]
    ]

    #Uses all index combinations of features sets and indexes allSets for combination
    for combo in combinations:
        feature_set = allSets[combo]
        svm = my_svm(X, Y, feature_set)
        X_, Y_ = svm.preprocess()
        svc = support_vector_machine.SVC()
        kf, tss = svm.cross_validation(svc, X_, Y_, k)
        # print(f'{feature_set}: {np.mean(tss)}')
        print(f'{feature_set}: {tss}')
        bestFold = float('-inf')
        cm = None
        tss = []
        
        # Performs K fold cross validation, calculates tss, and finds best fold to create confusion matrix for
        for train_index, test_index in kf.split(X_):
            # Indexes dataset for folds
            X_train, X_test = X_[train_index], X_[test_index]
            Y_train, Y_test = Y_[train_index], Y_[test_index]

            svc.fit(X_train, Y_train)
            # Gets confusion matrix and score for each fold
            confusion_matrix, score = svm.tss(svc, X_test, Y_test)
            tss.append(score)

            # Stores best fold and corresponding confusion matrix
            if score > bestFold:
                bestFold = score
                cm = confusion_matrix
        
        #Finds best feature set combination using highest mean tss score
        if np.mean(tss) > highest_tss:
            highest_tss = np.mean(tss)
            best_feature_set = feature_set
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))

        #Renders confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax1)
        ax1.set_title(f'Feature Set: {feature_set}')

        #Renders TSS scores
        ax2.plot(np.linspace(1, k, num=k), tss, color='r', label={f'Avg TSS Score: {np.mean(tss)}'})
        ax2.set_xlabel("K-fold")
        ax2.set_ylabel("TSS score")
        ax2.set_title(f'TSS-scores for Feature Set: {feature_set}')

        plt.tight_layout()
        plt.legend()
        plt.show()

    print(f'Best feature set combination based off average TSS score: {best_feature_set}')

# data_experiment() function executes 2 experiments with 2 data sets.
# svm is trained (and tested) on both 2010 data and 2020 data
# the output of this function is : 
#  1) TSS average scores for k-fold validation printed out on console.
#  2) Confusion matrix for TP, FP, TN, FN. See assignment document 
#  3) A chart showing TSS scores for all folds of CV. 
#     This means that for each fold, compute the TSS score on test set for that fold, and plot it.
#     The number of folds will decide the number of points on this chart (i.e. 10)
# above 3 charts are produced for both datasets
# feature set for this experiment should be the 
# best performing feature set combination from feature_experiment()
def data_experiment():
    best_feature_set = ['FS-I', 'FS-II', 'FS-IV']
    datasets = ['data-2010-15', 'data-2020-24']
    k = 10

    #loops through both datasets to render required graphs
    for d in datasets:
        X, Y = load_data(d)
        svm = my_svm(X, Y, best_feature_set)
        X_, Y_ = svm.preprocess()
        svc = support_vector_machine.SVC()
        kf, tss = svm.cross_validation(svc, X_, Y_, k)

        bestFold = float('-inf')
        cm = None
        tss = []

        # Performs K fold cross validation, calculates tss, and finds best fold to create confusion matrix for
        for train_index, test_index in kf.split(X_):
            X_train, X_test = X_[train_index], X_[test_index]
            Y_train, Y_test = Y_[train_index], Y_[test_index]

            svc.fit(X_train, Y_train)
            confusion_matrix, score = svm.tss(svc, X_test, Y_test)
            tss.append(float(score))

            if score > bestFold:
                bestFold = score
                cm = confusion_matrix
        
        print(f'{best_feature_set}: {tss}')
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))

        #Renders confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax1)
        ax1.set_title(f'Feature Set: {best_feature_set} on {d}')

        #Renders TSS scores
        ax2.plot(np.linspace(1, k, num=k), tss, color='r', label={f'Avg TSS Score: {np.mean(tss)}'})
        ax2.set_xlabel("K-fold")
        ax2.set_ylabel("TSS score")
        ax2.set_title(f'TSS-scores for Feature Set: {best_feature_set}')

        plt.tight_layout()
        plt.legend()
        plt.show()



# below should be your code to call the above classes and functions
# with various combinations of feature sets
# and both datasets

#Splits dataset into feature sets as a dictionary to be indexed when making feature set combinations.
def load_data(dataset):
    data_order = np.load(f'./{dataset}/data_order.npy')
    X = {}

    pfmt = np.load(f'./{dataset}/pos_features_main_timechange.npy', allow_pickle=True)
    nfmt = np.load(f'./{dataset}/neg_features_main_timechange.npy')
    #Orders data in the data_order given
    fs1 = np.concatenate((pfmt, nfmt))[data_order, :18]
    fs2 = np.concatenate((pfmt, nfmt))[data_order, 18:]
    X['FS-I'] = fs1
    X['FS-II'] = fs2

    pfh = np.load(f'./{dataset}/pos_features_historical.npy')
    nfh = np.load(f'./{dataset}/neg_features_historical.npy')
    fs3 = np.concatenate((pfh, nfh))[data_order, :]
    X['FS-III'] = fs3
    # fs3 is just 0's

    pfm = np.load(f'./{dataset}/pos_features_maxmin.npy')
    nfm = np.load(f'./{dataset}/neg_features_maxmin.npy')
    fs4 = np.concatenate((pfm, nfm))[data_order, :]
    X['FS-IV'] = fs4

    pos = np.load(f'./{dataset}/pos_class.npy')
    neg = np.load(f'./{dataset}/neg_class.npy', allow_pickle=True)
    #based on harp number
    Y = np.concatenate((pos, neg))[data_order, :]

    return X, Y


feature_experiment()

print('--------------------')
data_experiment()
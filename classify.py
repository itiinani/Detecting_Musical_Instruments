
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import sklearn as sk
import librosa
import glob
import os
from sklearn import svm

from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import neighbors
from sklearn import decomposition
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import precision_recall_fscore_support
#import Preprocessing


def extract_for_one(filename):
    loader = essentia.standard.MonoLoader(filename = filename)
    features = FeatureExtractor(frameSize=2048, hopSize=1024, sampleRate=44100)
    p = essentia.Pool()
    for desc, output in features.outputs.items():
        output >> (p, desc)

    essentia.run(loader)

    stats = ['mean', 'var', 'dmean', 'dvar']
    statsPool = essentia.standard.PoolAggregator(defaultStats=stats)(p)
    return statsPool


# In[16]:

def convert_pool_to_dataframe(essentia_pool, inst_code, filename):
    pool_dict = dict()
    for desc in essentia_pool.descriptorNames():
        if type(essentia_pool[desc]) is float:
            pool_dict[desc] = essentia_pool[desc]
        elif type(essentia_pool[desc]) is numpy.ndarray:
            # we have to treat multivariate descriptors differently
            for i, value in enumerate(essentia_pool[desc]):
                feature_name = "{desc_name}{desc_number}.{desc_stat}".format(
                    desc_name=desc.split('.')[0],
                    desc_number=i,
                    desc_stat=desc.split('.')[1])
                pool_dict[feature_name] = value
    pool_dict['inst_code'] = inst_code
    return pd.DataFrame(pool_dict, index=[os.path.basename(filename)])


# In[3]:

def preprocess_traindata():
    features = pd.DataFrame()
    #reading all files recursively
    files = glob.glob('Datasets\TrainingData\*\*.wav')
    np.random.shuffle(files)
    for filename in files:
        #Preprocess the music file to get features from it
        music, sr = librosa.load(filename)
        mfccs = librosa.feature.mfcc(y=music, sr=sr)
        mean_mfccs = np.mean(mfccs, axis = 1)
        feature = mean_mfccs.reshape(20)
        if '[cel]' in filename[25:38]:
            instrument_code = 1
        elif '[flu]' in filename[25:38]:
            instrument_code = 2
        elif '[gac]' in filename[25:38]:
            instrument_code = 3
        elif '[gel]' in filename[25:38]:
            instrument_code = 4
        elif '[org]' in filename[25:38]:
            instrument_code = 5
        elif '[pia]' in filename[25:38]:
            instrument_code = 6
        elif '[sax]' in filename[25:38]:
            instrument_code = 7
        elif '[tru]' in filename[25:38]:
            instrument_code = 8
        elif '[vio]' in filename[25:38]:
            instrument_code = 9
        elif '[cla]' in filename[25:38]:
            instrument_code = 10
        elif '[voi]' in filename[25:38]:
            instrument_code = 11
        else:
            instrument_code = 0
            print('Unknown instrument found in the file', filename)

        feature = extract_for_one(filename)
        features = features.append(convert_pool_to_dataframe(feature, instrument_code, filename))
    
    filename='features.csv'
    features.to_csv(filename, index=False)   
    return features
    

def preprocess_trainingdata(data):    
    y = data['inst_code']
    X = data.drop(['inst_code'], axis=1).values
    X = preprocessing.Imputer().fit_transform(X)

    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)

    estimators = [("scale", preprocessing.StandardScaler()),
                  ('anova_filter', SelectKBest(mutual_info_classif, k=100)),
                  ('svm', svm.SVC(decision_function_shape='ovo'))]
    
    clf = Pipeline(estimators)

    return clf, X,y

def train_and_test(clf, X,y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    Evaluate_accuracy(y_test, y_pred)
    return clf

def Evaluate_accuracy(pred, true_value):
    #Evaluate the accuracy of the model
    print("Accuracy score is", accuracy_score(true_value.astype(int), pred.astype(int))*100)
    print("Mean squared error", mean_squared_error(true_value, pred))
    rmse = np.sqrt(mean_squared_error(true_value, pred))
    print("Root Mean Squared Error: {}".format(rmse))
    print("Mean absolute error:", mean_absolute_error(true_value,pred))
    print("Classification Report: ",classification_report(true_value, pred))
    print('confusion matrix:', confusion_matrix(true_value, pred))
    print "Micro stats:"
    print precision_recall_fscore_support(true_value, pred, average='micro')
    print "Macro stats:"
    print precision_recall_fscore_support(true_value, pred, average='macro')
    return


def main():
    features = preprocess_traindata()

    model, features, targetvalue= preprocess_trainingdata(features)
    clf = train_and_test(model, features, targetvalue)


if __name__ == '__main__':
    main()


# On testing with training data itself:
# When using KNN:
# ('Accuracy score is', 62.4750499001996)
# ('Mean squared error', 7.3732534930139719)
# Root Mean Squared Error: 2.71537354576
# ('Mean absolute error:', 1.4131736526946108)
# When using SVM:
# ('Number of filessssssssssssss trained', 1001)
# ('Accuracy score is', 14.685314685314685)
# ('Mean squared error', 31.002997002997002)
# Root Mean Squared Error: 5.56803349514
# ('Mean absolute error:', 4.615384615384615)
# 
# when using this code:
#     X = preprocessing.Imputer().fit_transform(features)
#     le = preprocessing.LabelEncoder()
#     y = le.fit_transform(instrument_code)
#     estimators = [("scale", preprocessing.StandardScaler()),
#                   ('svm', svm.SVC(decision_function_shape='ovo'))]
#     
#     clf = Pipeline(estimators)
#     params = dict(svm__kernel=['rbf'], svm__C=[0.1],
#                   svm__degree=[1, 3], svm__gamma=[0.01])
#     gs = GridSearchCV(clf, param_grid=params, cv=10, verbose=2)
#     gs.fit(X, y)
#     prediction = gs.predict(X)
#     
# ('Accuracy score is', 38.031319910514547)
# ('Mean squared error', 12.879940343027592)
# Root Mean Squared Error: 3.58886337759
# ('Mean absolute error:', 2.410439970171514)
# ('Classification Report: ', '             precision    recall  f1-score   support\n\n          0       0.78      0.08      0.14       388\n          1       0.56      0.07      0.12       451\n          2       0.40      0.52      0.45       637\n          3       0.35      0.44      0.39       760\n          4       0.37      0.56      0.45       682\n          5       0.41      0.56      0.47       721\n          6       0.30      0.23      0.26       626\n          7       0.51      0.27      0.35       577\n          8       0.45      0.28      0.35       580\n          9       0.51      0.19      0.28       505\n         10       0.33      0.61      0.42       778\n\navg / total       0.43      0.38      0.35      6705\n')
# ('confusion matrix:', array([[ 31,   2,  89,  52,  19,  58,  36,   1,  63,   4,  33],
#        [  2,  31,  53,  21,  98,  86,  21,  11,  27,  27,  74],
#        [  0,   0, 333,  44,  66,  60,  19,   4,   4,   0, 107],
#        [  1,   1,  37, 337,  81,  56,  25,   6,  15,   0, 201],
#        [  1,   0,  30,  96, 380,  13,   5,  15,   4,   2, 136],
#        [  1,   1,  42,  43, 107, 402,  43,  11,   5,   4,  62],
#        [  0,   3,  51,  71,  50,  87, 146,  63,  13,  15, 127],
#        [  2,   5,  16,  55,  62,  70,  52, 156,  26,  27, 106],
#        [  1,   5,  61,  80,  37,  53,  64,  10, 165,  11,  93],
#        [  1,   7,  72,  47,  22,  96,  67,  22,  37,  96,  38],
#        [  0,   0,  50, 114, 102,   8,  13,   6,   9,   3, 473]]))
# Micro stats:
# (0.38031319910514544, 0.38031319910514544, 0.38031319910514544, None)
# Macro stats:
# (0.45071493277784475, 0.34688204167794512, 0.33565231678868362, None)


# coding: utf-8

# In[7]:

import numpy as np
import pandas as pd
import sklearn as sk
from feature_extraction import extract_for_one
from feature_extraction import convert_pool_to_dataframe
from sklearn import neighbors
import sys
from sklearn.externals import joblib
from sklearn import preprocessing
IRMAS = 1
PHILHARMONIA = 2
MIS =3


# In[8]:

def preprocess(data):
    y_test = data['class']
    X_test = data.drop(['class'], axis=1).values
    X_test = preprocessing.Imputer().fit_transform(X_test)
    return X_test, y_test


# In[9]:

def loadModel(dataset):
    if dataset ==IRMAS:
        model = joblib.load('irmas.model')
    if dataset == PHILHARMONIA:
        model = joblib.load('Philharmonia.model')
    if dataset == MIS:
        model = joblib.load('mis.model')
    return model


# In[10]:

def predict(model, features, instrument):
    print("Testing the model")
    prediction = model.predict(features)
    return prediction


# In[22]:

def print_prediction(prediction1, prediction2, prediction3, instrument):
    
    dict = {}
    dict[0]='Unknown instrument'
    dict[1]='Cello'
    dict[2]='Saxophone'
    dict[3]='Clarinet'
    dict[4]='Flute'
    dict[5]='Guitar'
    dict[6]='Piano'
    dict[7]='Trumpet'
    dict[8]='Violin'
    dict[9]='Banjo'
    dict[10]='Mandolin'
    dict[11]='Organ'
    dict[12]='Acoustic Guitar'
    dict[13]='Electric Guitar'
    dict[14]='Voice'
    finalprediction = ''
    print("The true instrument in the music piece is ", dict[instrument])
    if (prediction1 == prediction2 and prediction1==prediction3):
        print('The instruments in the music piece is predicted as ', dict[prediction1])
        finalprediction = dict[prediction1]
    elif prediction1 == prediction2 or prediction2==prediction3:
        print('The instruments in the music piece is predicted as ', dict[prediction2])
        finalprediction = dict[prediction2]
    elif prediction1 == prediction3 or prediction2==prediction3:
        print('The instruments in the music piece is predicted as ', dict[prediction3])
        finalprediction = dict[prediction3]
    elif prediction1 == prediction2 or prediction3 == prediction1:
        print('The instruments in the music piece is predicted as ', dict[prediction1])
        finalprediction = dict[prediction1]
    elif prediction1 == 0 and prediction3 == 0 and prediction2!=0:
        print('The instruments in the music piece is predicted as ', dict[prediction2])
        finalprediction = dict[prediction2]
    elif prediction2 == 0 and prediction3 == 0 and prediction1!=0:
        print('The instruments in the music piece is predicted as ', dict[prediction1])
        finalprediction = dict[prediction1]
    elif prediction2 == 0 and prediction1 == 0 and prediction3!=0:
        print('The instruments in the music piece is predicted as ', dict[prediction3])
        finalprediction = dict[prediction3]
    else:
        #irmas model is built by training huge music files. So choosing the prediction of irmas model
        print('Prediction 1:', dict[prediction1])
        finalprediction = dict[prediction1]
    print(dict[prediction1],dict[prediction2],dict[prediction3],finalprediction)
    with open("static/Output.txt","w") as text_file:
           text_file.write(dict[prediction1])
    return


# In[32]:

def fileDetection(filename):
    print("Enter the audio file name you want to test:")
    dir_name='Datasets/TestingData/'
    base_filename= filename
    filename = dir_name+base_filename
    print filename
    print("Get features from the music file")
    instrument_code = 1
    feature_set = pd.DataFrame()
    filename = str(filename)
    feature = extract_for_one(filename)
    features = feature_set.append(convert_pool_to_dataframe(feature, instrument_code, filename))
    features,instrument = preprocess(features.head(1))
    model1 = loadModel(IRMAS)
    prediction1 = predict(model1, features, instrument)
    
    model2 = loadModel(PHILHARMONIA)
    prediction2 = predict(model2, features, instrument)
    
    model3 = loadModel(MIS)
    prediction3 = predict(model3, features, instrument)

    print_prediction(prediction1[0], prediction2[0], prediction3[0], instrument[0])


# In[33]:

if __name__ == '__main__':
    main()


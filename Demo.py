
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import sklearn as sk
import librosa as lib
from sklearn import neighbors
import sys
from sklearn.externals import joblib
from sklearn import preprocessing
IRMAS = 1
RWC=2
PHILHARMONICA = 3


# In[36]:

def preprocess(data):

    y_test = data['class']
    #le = preprocessing.LabelEncoder()
    #y_test = le.fit_transform(y_test)
    X_test = data.drop(['class'], axis=1).values
    X_test = preprocessing.Imputer().fit_transform(X_test)
    print("class ::::", y_test)
    return X_test, y_test


# In[3]:

def loadModel(dataset):
    if dataset ==IRMAS:
        model = joblib.load('irmas1.model')
    if dataset == RWC:
        model = joblib.load('rwc1.model')
    if dataset == PHILHARMONICA:
        model = joblib.load('Philharmonica1.model')
    return model


# In[25]:

def predict(model, features, instrument):
    print("Testing the model")
    prediction = model.predict(features)
    print("The instrument actually is :", instrument)
    return prediction


# In[43]:

def main():
    print("Enter the audio file name you want to test:")
    dir_name='Datasets/TestingData-Part1/'
    base_filename= input()
    filename = dir_name+base_filename
    print(filename)
    print("Get features from the music file")
    #feature = extract_for_one(filename)
    #features = feature_set.append(convert_pool_to_dataframe(feature, instrument_code, filename))
    feature = pd.DataFrame()
    feature = pd.read_csv('Datasets/philharmoni/phil_features_cel.csv')
    features, instrument= preprocess(feature.head(1))

    model1 = loadModel(IRMAS)
    prediction1 = predict(model1, features, instrument)
    
    #model2 = loadModel(RWC)
    #prediction2 = predict(model2, features, instrument)
    prediction2 = 0
    model3 = loadModel(PHILHARMONICA)
    prediction3 = predict(model3, features, instrument)
    
    print(" predictions of the music instrument are: ", prediction1, prediction2, prediction3)
    
    dict = {}
    dict['0']='unknown'
    dict['1']='Piano'
    dict['2']='Electric Piano'
    dict['3']='Harpsichord Piano'
    dict['4'] = 'Glockenspiel'
    dict['5']='Marimba'
    dict['6']='Organ'
    dict['7']='Accordion'
    dict['8']='Harmonica'
    dict['9']='Guitar'
    dict['10']='Ukulele Guitar'
    dict['11']='Acoustic Guitar'
    dict['12']='Mandolin'
    dict['13']='Electric Guitar'
    dict['14']='Electric Bass'
    dict['15']='Violin'
    dict['16']='Viola'
    dict['17']='Cello'
    dict['18']='Contrabass'
    dict['19']='Harp'
    dict['20']='Timpani'
    dict['21']='Trumpet'
    dict['22']='Trombone'
    dict['23']='Tuba'
    dict['24']='Horn'
    dict['25']='Soprano Sax'
    dict['26']='Alto Sax'
    dict['27']='Tenor Sax'
    dict['28']='Baritone Sax'
    dict['29']='English Horn'
    dict['30']='Bassoon'
    dict['31']='Clarinet'
    dict['32']='Piccolo'
    dict['33']='Flute'
    dict['34']='Recorder'
    dict['35']='Shakuhachi'
    dict['36']='Banjo'
    dict['37']='Shamisen'
    dict['38']='Koto'
    dict['39']='Sho'
    dict['40']='Japanese Percussion'
    dict['41']='Drums'
    dict['42']='Rock Drums'
    dict['43']='Jazz Drums'
    dict['44']='Percussion'
    dict['45']='Soprano'
    dict['46']='Alto'
    dict['47']='Tenor'
    dict['48']='Baritone'
    dict['49']='Bass'
    dict['50']='R&B'
    dict['51']='voice'
    """
    if (prediction1 == prediction2 and prediction1==prediction3):
        print('The instruments in the music piece is predicted as ', dict[prediction1])
    elif prediction1 == prediction2 or prediction2==prediction3:
        print('The instruments in the music piece is predicted as ', dict[prediction2])
    elif prediction1 == prediction3 or prediction2==prediction3:
        print('The instruments in the music piece is predicted as ', dict[prediction3])
    elif dict[prediction1] in dict[prediction2] or dict[prediction3] in dict[prediction2]:
        print('The instruments in the music piece is predicted as ', dict[prediction2])
    elif prediction1 == 0 and prediction3 == 0 and prediction2!=0:
        print('The instruments in the music piece is predicted as ', dict[prediction2])
    elif prediction2 == 0 and prediction3 == 0 and prediction1!=0:
        print('The instruments in the music piece is predicted as ', dict[prediction1])
    elif prediction2 == 0 and prediction1 == 0 and prediction3!=0:
        print('The instruments in the music piece is predicted as ', dict[prediction3])
    else:
    """
    print('Model 1: The instruments in the music piece is predicted as', dict[prediction1])
    #print('Model 2: The instruments in the music piece is predicted as', dict[prediction2])
    print('Model 3: The instruments in the music piece is predicted as', dict[prediction3])


# In[44]:

if __name__ == '__main__':
    main()



# coding: utf-8

# In[9]:

import numpy as np
import sklearn as sk
import librosa
import glob
import os
import pandas as pd


# In[19]:

def preprocess():
    music_data = []
    filelist = []
    featurelist = []
    inst_code = []
    #reading all files recursively
    files = glob.glob('Datasets\TestingData-Part1\Part1\*.wav')
    for filename in files:
        music, sr = librosa.load(filename)
        mfccs = librosa.feature.mfcc(y=music, sr=sr)
        mean_mfccs = np.mean(mfccs, axis = 1)
        feature = mean_mfccs.reshape(20)
        instrument_file = os.path.splitext(filename)[0]+'.txt'
        f = open(instrument_file, 'r')
        instrument = []
        for word in f:
            if 'cel' in word:
                instrument_code = 1
            elif 'cla' in word:
                instrument_code = 2
            elif 'flu' in word:
                instrument_code = 3
            elif 'gac' in word:
                instrument_code = 4
            elif 'gel' in word:
                instrument_code = 5
            elif 'org' in word:
                instrument_code = 6
            elif 'pia' in word:
                instrument_code = 7
            elif 'sax' in word:
                instrument_code = 8
            elif 'tru' in word:
                instrument_code = 9
            elif 'vio' in word:
                instrument_code = 10
            elif 'voi' in word:
                instrument_code = 11
            else:
                instrument_code = 0
            instrument.append(instrument_code)
        filelist.append(filename)
        featurelist.append(feature)
        inst_code.append(instrument)
    df=pd.DataFrame()    
    df['filename']= filelist
    df['feature']= featurelist
    df['instrumentcode']= inst_code
    print df.head(5)
    #file_data = [filename, feature, instrument]
    #music_data.append(file_data)
    #print file_data
    #return music_data
    return df


# In[3]:

def main():
    music_data = preprocess()
    print music_data


# In[20]:

if __name__ == '__main__':
     main()


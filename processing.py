import csv
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy as sc
import numpy as np

v = TfidfVectorizer(analyzer='char')


# Processing the test set
testsetx = pd.read_csv('data/test_set_x.csv')
testsetx['Text'] = testsetx['Text'].str.replace("[^\u0041-\u005a\u0061-\u007a\u0080-\u024f]+", "")
# testsetx['Text'] = testsetx['Text'].str.replace("[\d]", "")
testsetx = testsetx.fillna(value="")
mtest = v.fit_transform(testsetx['Text']).toarray()

for i, col in enumerate(v.get_feature_names()):
    testsetx[col] = mtest[:, i]


testsetx.drop('Id', axis=1, inplace=True)
testsetx.drop('Text', axis=1, inplace=True)


# Processing the training set
trainsetx = pd.read_csv('data/train_set_x.csv')
trainsetx['Text'] = trainsetx['Text'].str.replace("[^\u0041-\u005a\u0061-\u007a\u0080-\u024f]+", "")
# trainsetx['Text'] = trainsetx['Text'].str.replace("[\d]", "")
trainsety = pd.read_csv('data/train_set_y.csv')
trainsetx['label'] = trainsety['Category']
trainsetx = trainsetx.dropna(axis=0, how='any')

mtrain = v.fit_transform(trainsetx['Text']).toarray()

for i, col in enumerate(v.get_feature_names()):
    trainsetx[col] = mtrain[:, i]

trainsetx.drop('Id', axis=1, inplace=True)
trainsetx.drop('Text', axis=1, inplace=True)


#pruning the columns
features = testsetx.columns[:-1]
for column in trainsetx:
    if column not in features and column is not 'label':
        trainsetx.drop(column,axis=1,inplace=True)
features = trainsetx.columns[:-1]
for column in testsetx:
    if column not in features and column is not 'label':
        testsetx.drop(column,axis=1,inplace=True)

# Showing the data
print(len(trainsetx.columns))
print(len(testsetx.columns))
print(trainsetx)
print(testsetx)
# Exporting the sets
trainsetx.to_csv('trainsetprocessed.csv')
testsetx.to_csv('testsetprocessed.csv')

from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import time

np.random.seed(0)


# print(clf.predict_proba(test[features])[0:10])
def loadTrainingDataset(filename, split):
    df = pd.read_csv(filename,header=0)
    df.drop('Unnamed: 0', axis=1, inplace=True)
    df['is_train'] = np.random.uniform(0, 1, len(df)) <= split
    trainingSet, testSet = df[df['is_train']==True], df[df['is_train']==False]
    return trainingSet, testSet,df

def loadTestingDataset(filename):
    df = pd.read_csv(filename,header=0)
    df.drop('Unnamed: 0', axis=1, inplace=True)
    return df


def trainModel(df,features):
    y = df['label']
    clf = RandomForestClassifier(n_estimators=100,n_jobs=-1, random_state=0)
    clf.fit(df[features], y)
    return clf

# finaldf = pd.DataFrame(clf.predict(test[features]))
# finaldf.to_csv('results.csv')

def getAccuracy(testArray, result):
    correct = 0
    for i in range(len(testArray)):
        if testArray[i] == result[i]:
            correct += 1
    return (correct/float(len(testArray))) *100.0

def main():
    split = 0.67
    trainingSet, testSet, fullSet = loadTrainingDataset('trainsetprocessed.csv', split)
    print('Train: ' + repr(len(trainingSet)))
    print('Test: ' + repr(len(testSet)))


    features = trainingSet.columns[1:-2]
    # Cross validated set
    t = time.time()
    model = trainModel(trainingSet,features)
    print(np.round_(time.time() - t, 3), 'sec taken for training')
    result = model.predict(testSet[features])
    testArray = testSet['label'].as_matrix()
    accuracy = getAccuracy(testArray,result)
    print('Accuracy: ' + repr(accuracy) + '%')


    # print("Training Model...")
    # t = time.time()
    # model = trainModel(fullSet,features)
    # print("Finished Training")
    # print(np.round_(time.time() - t, 3), 'sec taken for training')
    #
    # testingSet = loadTestingDataset('testsetprocessed.csv')
    # print("Making Predictions...")
    # result = model.predict(testingSet)
    # print("Finished Predictions")
    #
    # output = pd.DataFrame(data=result)
    # output.to_csv('output5000trees2.csv')


main()

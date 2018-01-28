from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
import time

np.random.seed(5461)


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
    clf = MLPClassifier(solver='lbfgs', alpha=1e-7, early_stopping=False,
                        hidden_layer_sizes=(30), random_state=1,
                        shuffle=True, activation='relu')
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


    t = time.time()
    features = trainingSet.columns[1:-2]
    # Cross validated set
    # model = trainModel(trainingSet,features)
    # result = model.predict(testSet[features])
    # testArray = testSet['label'].as_matrix()
    # accuracy = getAccuracy(testArray,result)
    # print('Accuracy: ' + repr(accuracy) + '%')

    model = trainModel(fullSet,features)

    testingSet = loadTestingDataset('testsetprocessed.csv')
    result = model.predict(testingSet)
    print(result)
    print(np.round_(time.time() - t, 3), 'sec elapsed')

    output = pd.DataFrame(data=result)
    output.to_csv('outputNNet30.csv')


main()

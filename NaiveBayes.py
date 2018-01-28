import pandas as pd
import numpy as np
import time
import math

np.random.seed(0)

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

def getAccuracy(testArray, result):
    correct = 0
    for i in range(len(result)):
        if testArray[i] == result[i]:
            correct += 1
    return (correct/float(len(testArray))) *100.0


def main():
    split = 0.67
    trainingSet, testSet, fullSet = loadTrainingDataset('trainsetprocessed.csv', split)
    print('Train: ' + repr(len(trainingSet)))
    print('Test: ' + repr(len(testSet)))
    t = time.time()
    print('Start training...')

    #Store each set of row vectors by class
    class0 = trainingSet.loc[trainingSet['label'] == 0]
    class1 = trainingSet.loc[trainingSet['label'] == 1]
    class2 = trainingSet.loc[trainingSet['label'] == 2]
    class3 = trainingSet.loc[trainingSet['label'] == 3]
    class4 = trainingSet.loc[trainingSet['label'] == 4]

    #Table to store probabilites in
    probabilities = pd.DataFrame(0, index=np.arange(5), columns=trainingSet.columns)
    probabilities['label'] = list(range(0, 5))
    #Calculate the Prior Probability P(c) for each class
    priors = [len(class0)/len(trainingSet), len(class1)/len(trainingSet),
              len(class2)/len(trainingSet), len(class3)/len(trainingSet),
              len(class4)/len(trainingSet)]
    probabilities['priors'] = priors



    #Calculate the probability of each character by langauge
    #smoothing value alpha
    alpha = 1

    for column in trainingSet.columns:
        if column == 'label' or column == 'is_train':
            continue
        #Calculate the multinomial values on each point
        probability = [(sum(class0[column])+alpha)/(sum(trainingSet[column])+5*alpha),
                       (sum(class1[column])+alpha)/(sum(trainingSet[column])+5*alpha),
                       (sum(class2[column])+alpha)/(sum(trainingSet[column])+5*alpha),
                       (sum(class3[column])+alpha)/(sum(trainingSet[column])+5*alpha),
                       (sum(class4[column])+alpha)/(sum(trainingSet[column])+5*alpha)]
        probabilities[column] = probability

    #Now we use max(P(c)) + sum(P(t|c)) to make predicitons
    print("Training complete")
    print("Running cross validation now. This may take a while depending on your machine.")
    #classification on test set
    classifcation = []
    for row in testSet.itertuples():
        scores = [-100,-100,-100,-100,-100]
        for case in range(0,5):
#            scores[case] = (math.log(probabilities['priors'][case]))
            scores[case] = (probabilities['priors'][case])
#            print(scores)
            #don't need +1 because we don't want to use 'is_train'
            for column in range(0,len(testSet.columns)):
                if column == 0 or column == 115:
                    continue
                scores[case] += row[column]*probabilities[probabilities.columns[column]][case]


        classifcation.append(scores.index(max(scores)))
#        count = count+1
        scores.clear()
    pint("Getting accuracy")
    testArray = testSet['label'].as_matrix()
    accuracy = getAccuracy(testArray,classifcation)
    print('Accuracy: ' + repr(accuracy) + '%')
    print(np.round_(time.time() - t, 3), 'sec elapsed')

main()

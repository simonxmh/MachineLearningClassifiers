from sklearn.svm import LinearSVC
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

    # GridSearch in order to tune the C parameter
    # Highest validation accuracy with C=300

    #clf=GridSearchCV(LinearSVC(random_state=0,multi_class='ovr'),
    #param_grid=[{'C':[0.01,0.1,10,100,300]}])

    # Setting the value of C to 300 will take 2-3 min to run the program
    # Setting it to e.g. 0.1 will take seconds to run

    print("Process takes 2-3 min to run when C value is 300")
    
    clf = LinearSVC(C=300, dual=True, multi_class='ovr',
                    penalty='l2',random_state=0,loss='hinge')

    clf.fit(df[features], y)

#    #Other options available

#    LinearSVC(C=1.5, class_weight=None, dual=True, fit_intercept=True,
#     intercept_scaling=1, loss='hinge', max_iter=10000,
#     multi_class='ovr', penalty='l2', random_state=0, tol=0.0001,
#     verbose=0)
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
    model = trainModel(trainingSet,features)
    result = model.predict(testSet[features])
    testArray = testSet['label'].as_matrix()
    accuracy = getAccuracy(testArray,result)
    print('Accuracy: ' + repr(accuracy) + '%')

    #print('Best score for data1:', model.best_score_)

    #print('Best C:', model.best_estimator_.C)

    # model = trainModel(fullSet,features)
    #
    # testingSet = loadTestingDataset('testsetprocessed.csv')
    # result = model.predict(testingSet)
    # print(result)
    print(np.round_(time.time() - t, 3), 'sec elapsed')

    # output = pd.DataFrame(data=result)
    # output.to_csv('output1000trees.csv')


main()

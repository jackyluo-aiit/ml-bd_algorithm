from math import log


def calEnt(dataSet):
    numOfEntries = len(dataSet)
    labelCount = {}
    entropy = 0.0
    for each in dataSet:
        currentLabel = each[-1]
        if currentLabel not in labelCount.keys():
            labelCount[currentLabel] = 0
        labelCount[currentLabel] += 1
    for each in labelCount.keys():
        prob = float(labelCount[each] / numOfEntries)
        entropy += prob * log(prob, 2)
    return -entropy


def createDataSet():
    dataSet = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    labels = ['nosurfacing', 'flippers']
    return dataSet, labels




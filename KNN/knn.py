from numpy import *
import operator  # 运算符模块


# import matplotlib


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classifier(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    squareDiff = diffMat ** 2
    sqDistance = sum(squareDiff, axis=1)
    distance = sqDistance * 0.5
    sortedDistance = argsort(distance)
    classCount = {}
    for i in range(k):
        voteLabel = labels[sortedDistance[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
        # count the occupancy of each class within top k
        # data points
        # near the input point
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]  # return the first field within the first element


def file2matrix(filename):
    with open(filename, 'r', encoding='utf8') as f:
        lines = f.readlines()
        fileLen = len(lines)
        returnMat = zeros((fileLen, 3))
    classLabelVector = []
    index = 0
    classLabels = {'didntLike': 1, 'smallDoses': 2, 'largeDoses': 3}
    for each in lines:
        listFromLine = each.strip().split('\t')
        returnMat[index, :] = listFromLine[0:3]
        try:
            classLabelVector.append(classLabels[listFromLine[-1]])  # 将字符串自动变成整型
        except:
            classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


def autoNorm(dataSet):
    minValue = dataSet.min(0)
    maxValue = dataSet.max(0)
    ranges = maxValue - minValue
    returnMat = zeros_like(dataSet)
    returnMat = dataSet - tile(minValue, (dataSet.shape[0], 1))
    returnMat = returnMat / tile(ranges, (returnMat.shape[0], 1))
    return returnMat, ranges, minValue


def datingClassTest():
    validate_rate = 0.1
    dateMat, dateLabel = file2matrix('KNN/datingTestSet.txt')
    normMat, ranges, minValue = autoNorm(dateMat)
    dataSize = normMat.shape[0]
    numOfTest = int(dataSize * validate_rate)
    errorCount = 0
    for i in range(numOfTest):
        result = classifier(normMat[i, :], normMat[numOfTest:dataSize, :], dateLabel[numOfTest:dataSize], 3)
        print("the result show that: %d; and the actual is: %d" % (result, dateLabel[i]))
        if result != dateLabel[i]:
            errorCount += 1
    print("the total error rate is: %f" % (errorCount / float(numOfTest)))


if __name__ == '__main__':
    datingClassTest()

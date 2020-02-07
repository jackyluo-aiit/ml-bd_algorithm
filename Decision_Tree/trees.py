import operator

from Decision_Tree.dataSet_Entropy import calEnt


def createDataSet():
    dataSet = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    labels = ['nosurfacing', 'flippers']
    return dataSet, labels


def splitDataSet(dataSet, axis, value):
    # 创建新list对象
    retDataSet = []
    for each in dataSet:
        if each[axis] == value:
            reducedEach = each[:axis]  # 包头不包尾
            reducedEach.extend(each[axis + 1:])
            retDataSet.append(reducedEach)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    numOfFeature = len(dataSet[0]) - 1
    baseEntropy = calEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numOfFeature):
        featList = [example[i] for example in dataSet]
        uniqFeats = set(featList)
        newEntropy = 0.0
        for value in uniqFeats:
            subDataSet = splitDataSet(dataSet, i, value)
            subEntropy = calEnt(subDataSet)
            newEntropy += len(subDataSet) / float(len(dataSet)) * subEntropy
        infoGain = baseEntropy - newEntropy
        print("%d feature's infoGain:%f" % (i, infoGain))
        if infoGain >= bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    print("best feature:", bestFeature)
    return bestFeature


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount


dataSet, labels = createDataSet()


def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]  # 将dataSet中所有element的类别取出
    if classList.count(classList[0]) == len(classList):  # 判断一种类别的数目是否等于整个List的总数，若相等就相当于只有一种类别
        return classList[0]  # 返回List中的类别
    if len(dataSet[0]) == 1:  # 判断进来的dataSet的第一个（任意一个）element中的element的数量是否只有一个，若是1个则证明已经遍历完特征
        return majorityCnt(classList)  # 返回此时dataSet中，占数目最多的类别
    bestFeatIndex = chooseBestFeatureToSplit(dataSet)  # 得到最佳的特征的index
    bestFeatLabel = labels[bestFeatIndex]
    returnTree = {bestFeatLabel: {}}  # 创建要返回的树
    del (labels[bestFeatIndex])  # 从特征表中移除该特征
    featValues = [example[bestFeatIndex] for example in dataSet]  # 取出该特征下的所有值
    uniqValues = set(featValues)
    for each in uniqValues:  # 遍历该特征下的所有值
        subLabels = labels[:]
        returnTree[bestFeatLabel][each] = createTree(splitDataSet(dataSet, bestFeatIndex, each), subLabels)  # 递归进行创建
    return returnTree


if __name__ == '__main__':
    dataSet, labels = createDataSet()
    print(createTree(dataSet, labels))

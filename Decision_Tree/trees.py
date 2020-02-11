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


def createTree(dataSet, labels):
    workLabels = labels.copy()
    classList = [example[-1] for example in dataSet]  # 将dataSet中所有element的类别取出
    if classList.count(classList[0]) == len(classList):  # 判断一种类别的数目是否等于整个List的总数，若相等就相当于只有一种类别
        return classList[0]  # 返回List中的类别
    if len(dataSet[0]) == 1:  # 判断进来的dataSet的第一个（任意一个）element中的element的数量是否只有一个，若是1个则证明已经遍历完特征
        return majorityCnt(classList)  # 返回此时dataSet中，占数目最多的类别
    bestFeatIndex = chooseBestFeatureToSplit(dataSet)  # 得到最佳的特征的index
    bestFeatLabel = workLabels[bestFeatIndex]
    returnTree = {bestFeatLabel: {}}  # 创建要返回的树
    del (workLabels[bestFeatIndex])  # 从特征表中移除该特征
    featValues = [example[bestFeatIndex] for example in dataSet]  # 取出该特征下的所有值
    uniqValues = set(featValues)
    for each in uniqValues:  # 遍历该特征下的所有值
        subLabels = workLabels[:]
        returnTree[bestFeatLabel][each] = createTree(splitDataSet(dataSet, bestFeatIndex, each), subLabels)  # 递归进行创建
    return returnTree


def classifier(inputTree, labels, inputVect):
    workLabels = labels.copy()
    firstFeat = list(inputTree.keys())[0]
    secondLayer = inputTree[firstFeat]  # 相当于获得根据第一个特征分割后的结果
    firstFeatIndex = workLabels.index(firstFeat)  # 相当于获得inputVect里，决策树第一个特征所在的index
    for key in secondLayer.keys():
        if key == inputVect[firstFeatIndex]:  # 检查inputVect在firstFeatIndex的位置的值是否属于第一个特征分割后的某个子集
            if type(secondLayer[key]).__name__ == 'dict':  # 检查该分类下是否还有分割的子集
                workLabels.remove(firstFeat)
                return classifier(secondLayer[key], workLabels, inputVect)
            else:
                return secondLayer[key]


def storeTree(inputTree, filename='decisionTree.txt'):
    import pickle
    fw = open(filename, 'wb')
    pickle.dump(inputTree, fw)
    fw.close()


def loadTree(filename):
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)


if __name__ == '__main__':
    dataSet, featsLabels = createDataSet()
    print(featsLabels)
    tree = createTree(dataSet, featsLabels)
    # storeTree(tree)
    # tree = loadTree('decisionTree.txt')
    # print(featsLabels)
    print(tree)
    print(classifier(tree, featsLabels, [1, 1]))

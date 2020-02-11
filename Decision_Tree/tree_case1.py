import operator
from Decision_Tree.trees import *
from Decision_Tree.dataSet_Entropy import calEnt
import os


def train_tree(filename):
    trainData = []
    with open('lenses.txt', 'r') as f:
        for line in f.readlines():
            lineList = line.strip().split('\t')
            trainData.append(lineList)
    print(len(trainData))
    labels = ['age', 'prescript', 'astigmatic', 'tearRate']
    trainTree = createTree(trainData, labels)
    print(trainTree)
    storeTree(trainTree, filename)


if __name__ == '__main__':
    treeFile = 'lenses_tree.txt'
    if not os.path.exists(treeFile):
        train_tree(treeFile)
    tree = loadTree(treeFile)
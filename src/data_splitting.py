import numpy as np
import pandas as pd

def defaultSplit(df, trainFrac):
    train = df.sample(frac=trainFrac)
    valid = df.drop(train.index)

    return train, valid

def splitStratified(df, trainFrac, stratCol):
    trainParts = []
    validParts = []

    for _, group in df.groupby(stratCol):
        groupTrain, groupValid = defaultSplit(group, trainFrac=trainFrac)
        trainParts.append(groupTrain)
        validParts.append(groupValid)
    
    train = pd.concat(trainParts)
    valid = pd.concat(validParts)

    train
    valid

    return train, valid


def splitFolds(df, k):
    idxs = df.sample(frac=1).index.to_numpy()
    folds = np.array_split(idxs, k)

    pairs = []
    for f in folds:
        train = df.drop(f)
        valid = df.loc[f]

        pairs.append((train, valid))

    return pairs


def splitStratifiedFolds(df, stratCol, k):
    groupFolds = {}

    for groupName, group in df.groupby(stratCol):
        idxs = group.sample(frac=1).index.to_numpy()
        groupFolds[groupName] = np.array_split(idxs, k)

    pairs = []

    for foldNum in range(k):
        validIdxs = np.concatenate([groupFolds[g][foldNum] for g in groupFolds.keys()])

        valid = df.loc[validIdxs]
        train = df.drop(validIdxs)

        pairs.append((train, valid))

    return pairs

def splitData(df, trainPart=80, stratify=None, folds=None):
    if trainPart > 1:
        trainFrac = trainPart / 100

    if stratify and folds: out = splitStratifiedFolds(df, stratify, folds)
    elif stratify: out = splitStratified(df, trainFrac, stratify)
    elif folds: out = splitFolds(df, folds)
    else: out = defaultSplit(df, trainFrac)

    return out

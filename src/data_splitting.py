import random

random.seed(41)

def splitData(df, trainPart=80, stratify=None):
    if stratify is None:
        idxs = df.index
        random.shuffle(list(idxs))
        splitAt = len(idxs) * trainPart // 100
        return df.loc[:splitAt], df.loc[splitAt:]

    trainIdxs = []
    for value in df[stratify].unique():
        subset = df[df[stratify] == value]
        idxs = subset.index
        random.shuffle(list(idxs))
        splitAt = len(idxs) * trainPart // 100
        trainIdxs.extend(idxs[:splitAt])

    return df.loc[trainIdxs], df.drop(trainIdxs)


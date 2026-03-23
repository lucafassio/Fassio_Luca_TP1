import numpy as np

def mse(yReal, yPred):
    err = yPred - yReal
    return np.mean(err ** 2)


def rmse(yReal, yPred):
    return np.sqrt(mse(yReal, yPred))


def mae(yReal, yPred):
    err = np.abs(yPred - yReal)
    return np.mean(err)


def r2Score(yReal, yPred):
    ssRes = np.sum((yReal - yPred) ** 2)
    ssTot = np.sum((yReal - np.mean(yReal)) ** 2)

    if ssTot == 0:
        return 0.0

    return 1 - (ssRes / ssTot)
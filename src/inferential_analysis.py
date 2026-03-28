import numpy as np
import pandas as pd

def zTestMeans(x, y, zCritic=1.96):
    n1, n2 = len(x), len(y)
    m1, m2 = x.mean(), y.mean()
    s1, s2 = x.var(ddof=1), y.var(ddof=1) # ddof calcula varianza muestral

    standarError = np.sqrt(s1 / n1 + s2 / n2)
    z = (m2 - m1) / standarError
    s_pooled = np.sqrt((s1 + s2) / 2)
    cohens_d = (m2 - m1) / s_pooled
    reject = np.abs(z) > zCritic

    return {
        'mean_diff': m2 - m1,
        'z_stat': z,
        'cohen': cohens_d,
        'reject_H0': reject
    }


def chiSquaredTest(observed, expected):
    rowTotals = observed.sum(axis=1)
    colTotals = observed.sum(axis=0)
    grandTotal = observed.values.sum()

    for row in observed.index:
        for col in observed.columns:
            expected.loc[row, col] = (rowTotals[row] * colTotals[col]) / grandTotal

    chiObs = 0.0
    for row in observed.index:
        for col in observed.columns:
            o = observed.loc[row, col]
            e = expected.loc[row, col]
            chiObs += ((o - e) ** 2) / e
    
    return chiObs


def cramersV(chi2, observed):
    n = observed.values.sum()
    k = min(observed.shape)
    return np.sqrt(chi2 / (n * (k - 1)))
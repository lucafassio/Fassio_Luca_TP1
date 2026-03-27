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

def performanceMetrics(target_real, predictions):
    target_real = np.asarray(target_real, dtype=int)
    predictions = np.asarray(predictions, dtype=int)

    tp = np.sum((target_real == 1) & (predictions == 1))
    tn = np.sum((target_real == 0) & (predictions == 0))
    fp = np.sum((target_real == 0) & (predictions == 1))
    fn = np.sum((target_real == 1) & (predictions == 0))

    accuracy = (tp + tn) / len(target_real) if len(target_real) > 0 else 0.0

    # metricas de venta
    precision_venta = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_venta = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_venta = 2 * precision_venta * recall_venta / (precision_venta + recall_venta) if (precision_venta + recall_venta) > 0 else 0.0

    # metricas de renta
    precision_renta = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    recall_renta = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_renta = 2 * precision_renta * recall_renta / (precision_renta + recall_renta) if (precision_renta + recall_renta) > 0 else 0.0

    return {
        'accuracy': accuracy,

        'precision_venta': precision_venta,
        'recall_venta': recall_venta,
        'f1_venta': f1_venta,

        'precision_renta': precision_renta,
        'recall_renta': recall_renta,
        'f1_renta': f1_renta,

        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
    }

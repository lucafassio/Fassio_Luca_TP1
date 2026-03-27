import numpy as np
import pandas as pd

from data_splitting import splitData
from preprocessing import normalizeData
from models import LinearRegression
from metrics import performanceMetrics

def sigmoidNormalize(values):
    values = np.clip(values, -500, 500)
    return 1.0 / (1.0 + np.exp(-values))

def evaluateOnePair(df_ba, features, k, threshold):
    '''
    Evalua un par umbral - features con k-fold.
    '''
    
    foldMetrics = []
    folds = splitData(df_ba, stratify='mercado_real', folds=k)
    rawPredictions = np.full(df_ba.shape[0], np.nan, dtype=float)
    classPredictions = np.full(df_ba.shape[0], np.nan, dtype=float)

    for train, valid in folds:
        train = train.copy()
        valid = valid.copy()

        train, valid = normalizeData(train, valid)
        train['mercado_real'] = (train['mercado_real'] == 'venta').astype(int)
        valid['mercado_real'] = (valid['mercado_real'] == 'venta').astype(int)

        model = LinearRegression(train[features], train['mercado_real'], addBias=True)
        model.fitByInverse()

        predRaw = model.prediction(valid[features])
        # predRaw = sigmoidNormalize(predRaw)
        predClass = (predRaw >= threshold).astype(int)

        metrics = performanceMetrics(valid['mercado_real'], predClass)
        foldMetrics.append(metrics)

        for i, idx in enumerate(valid.index):
            pos = df_ba.index.get_loc(idx)
            rawPredictions[pos] = predRaw[i]
            classPredictions[pos] = predClass[i]

    foldMetrics = pd.DataFrame(foldMetrics)
    summary = {
        'accuracy_mean': foldMetrics['accuracy'].mean(),
        'accuracy_std': foldMetrics['accuracy'].std(ddof=1),
        
        'precision_venta_mean': foldMetrics['precision_venta'].mean(),
        'precision_venta_std': foldMetrics['precision_venta'].std(ddof=1),
        'recall_venta_mean': foldMetrics['recall_venta'].mean(),
        'recall_venta_std': foldMetrics['recall_venta'].std(ddof=1),
        'f1_venta_mean': foldMetrics['f1_venta'].mean(),
        'f1_venta_std': foldMetrics['f1_venta'].std(ddof=1),

        'precision_renta_mean': foldMetrics['precision_renta'].mean(),
        'precision_renta_std': foldMetrics['precision_renta'].std(ddof=1),
        'recall_renta_mean': foldMetrics['recall_renta'].mean(),
        'recall_renta_std': foldMetrics['recall_renta'].std(ddof=1),
        'f1_renta_mean': foldMetrics['f1_renta'].mean(),
        'f1_renta_std': foldMetrics['f1_renta'].std(ddof=1)
    }

    return summary

def evaluateCombinations(df_ba, featuresCombinations, k=10):
    
    thresholds = np.linspace(0.1, 0.9, 50)
    for name, config in featuresCombinations.items():
        bestMetricValue = -np.inf
        bestSummary = None
        bestThreshold = None

        for th in thresholds:
            out = evaluateOnePair(
                df_ba=df_ba,
                features=config['features'],
                k=k,
                threshold=th
            )

            currentMetricValue = out['f1_renta_mean']

            if currentMetricValue > bestMetricValue:
                bestMetricValue = currentMetricValue
                bestSummary = out
                bestThreshold = th

        featuresCombinations[name]['metric'] = bestSummary
        featuresCombinations[name]['best_threshold'] = bestThreshold

    return featuresCombinations

def printModelRanking(featuresCombinations, rankBy='f1_renta_mean'):
    rows = []

    for name, comb in featuresCombinations.items():
        m = comb['metric']
        rows.append({
            'modelo': name,
            'threshold': comb['best_threshold'],
            'accuracy_mean': m['accuracy_mean'],
            'f1_venta_mean': m['f1_venta_mean'],
            'recall_venta_mean': m['recall_venta_mean'],
            'f1_renta_mean': m['f1_renta_mean'],
            'recall_renta_mean': m['recall_renta_mean']
        })

    rankingDf = pd.DataFrame(rows)
    rankingDf = rankingDf.sort_values(rankBy, ascending=False).reset_index(drop=True)

    # solo me interesa printear el podio
    rankingToPrint = rankingDf.head(3).copy()

    print('Ranking de los 3 mejores modelos:')
    print(f'Metrica de interes para el ranking: {rankBy}')
    print()

    for i, row in rankingToPrint.iterrows():
        print(f'#{i + 1}  {row["modelo"]}')
        print(f'threshold:    {row["threshold"]:.3f}')
        print(f'accuracy:     {row["accuracy_mean"]:.4f}')
        print(f'f1_venta:     {row["f1_venta_mean"]:.4f}')
        print(f'recall_venta: {row["recall_venta_mean"]:.4f}')
        print(f'f1_renta:     {row["f1_renta_mean"]:.4f}')
        print(f'recall_renta: {row["recall_renta_mean"]:.4f}')
        print()

    best = rankingDf.iloc[0]
    return best
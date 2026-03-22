import unicodedata
import numpy as np
import pandas as pd

from utils import sqftToM2

def preprocessingData(df):
    """
    Healer del dataset, manipula el dataframe para arreglar outliers, errores y datos faltantes.
        Reglas de limpieza:
         - precio: se eliminan NaNs, valores negativos y propiedades gratuitas.
         - tipo: no se toca, los valores estan sanos.
         - Área: se chequea que todos los datos sean floats positivos. Se modifica el nombre a area.
         - metros cubiertos: se chequea que todos los datos sean floats positivos.
         - unidades: se elimina esta columna y pasamos las dos anteriores a m2 unicamente.
         - ambientes: se chequea que todos los datos sean ints positivos.
         - pisos: se chequea que todos los datos sean ints positivos.
         - pileta: se chequea que todos los datos sean bools.
         - lat, lon: se chequea que todos los datos sean floats.
         - edad: se chequea que todos los datos sean floats positivos.
    """
    df.columns = [
        unicodedata.normalize('NFKD', col).encode('ascii', 'ignore').decode('utf-8').lower() 
        for col in df.columns
    ]
    df = repairPrice(df)
    df = repairDimensions(df)
    df = repairRestOfData(df)
    return df

    
def repairPrice(df):
    df = df.dropna(subset=['precio'])
    df = df[df['precio'] > 0]
    df = df.reset_index(drop=True)
    return df


def repairDimensions(df):
    nans = [0, 0]
    for row in range(df.shape[0]):
        area = df['area'][row]
        mtsCov = df['metros_cubiertos'][row]

        if area <= 0 or area.dtype != float: 
            nans[0] += 1
        if mtsCov <= 0 or mtsCov.dtype != float: 
            nans[1] += 1

        if df['unidades'][row] == 'sqft':
            df.loc[row, 'area'] = sqftToM2(area)
            df.loc[row, 'metros_cubiertos'] = sqftToM2(mtsCov)
    
    df = df.drop(columns=['unidades'])
    print(f"Filas con area invalida: {nans[0]}")
    print(f"Filas con metros cubiertos invalidos: {nans[1]}")
    return df


def repairRestOfData(df):
    nans = [0, 0, 0, 0, 0, 0]
    types = [int, int, bool, float, float, float]
    keys = ['ambientes', 'pisos', 'pileta', 'lat', 'lon', 'edad']
    for key, type in zip(keys, types):
        for row in range(df.shape[0]):
            value = df[key][row]
            if (type == int and value <= 0) or pd.isna(value):
                nans[keys.index(key)] += 1
                continue

            if value.dtype != type:
                df.loc[row, key] = type(value)

    print(f"Filas con ambientes invalidos: {nans[0]}")
    print(f"Filas con pisos invalidos: {nans[1]}")
    print(f"Filas con pileta invalida: {nans[2]}")
    print(f"Filas con latitud invalida: {nans[3]}")
    print(f"Filas con longitud invalida: {nans[4]}")
    print(f"Filas con edad invalida: {nans[5]}")
    return df

def zScoreParams(df, feature):
    mu = df[feature].mean()
    sigma = df[feature].std()

    return mu, sigma

def zScoreApply(df, feature, mu, sigma):
    return (df[feature] - mu) / sigma

def normalizeData(train, valid):
    train['precio'] = np.log(train['precio'])
    valid['precio'] = np.log(valid['precio'])
    
    for type in train['tipo'].unique():
        head = 'es_' + type
        train[head] = (train['tipo'] == type).astype(int)
        valid[head] = (valid['tipo'] == type).astype(int)

    train = train.drop(columns=['tipo'])
    valid = valid.drop(columns=['tipo'])

    train['edad'] = np.log(train['edad'])
    valid['edad'] = np.log(valid['edad'])

    for feature in ['area', 'metros_cubiertos', 'edad']:
        mu, sigma = zScoreParams(train, feature)

        train[feature] = zScoreApply(train, feature, mu, sigma)
        valid[feature] = zScoreApply(valid, feature, mu, sigma)

    train['pileta'] = train['pileta'].astype(int)
    valid['pileta'] = valid['pileta'].astype(int)

    return train, valid

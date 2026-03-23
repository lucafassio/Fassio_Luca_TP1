import numpy as np


def getRandomSample(df, n):
    '''
    Devuelve un nuevo dataframe con n filas seleccionadas aleatoriamente del dataframe original.
        Args:
            df (pd.DataFrame): Dataset con los datos.
            n (int): Cantidad de filas a seleccionar.
        
        Returns:
            pd.DataFrame con las filas deseadas.
    '''
    return df.sample(n=n, random_state=42).reset_index(drop=True)


def countUniqueData(df, column):
    '''
    Muestra por terminal la cantidad de ocurrencias de cada valor unico en una columna.
        Args:
            df (pd.DataFrame): Dataset con los datos.
            column (str): Feature a analizar.
    '''
    counter = {}
    for value in df[column]:
        if counter.get(value): continue
        counter[value] = len(df[df[column] == value])
    
    print(f"Datos de la columna '{column}':")
    for key, value in counter.items():
        print(f' - {key}: {value}')


def sqftToM2(x):
    return x * 0.092903

def cutBetweenBells(array, valleyRange, bins=30):
    '''
    Recibe un array de datos y busca el valor que separa dos campanas de una distribucion bimodal en un rango estimado.
        Args:
            array (np.array): Datos a analizar.
            valleyRange (list): Limites izquierdo y derecho del rango que se desea buscar el valle.
        
        Returns:
            cutValue (np.float64): Valor medio de los extremos del bin que contiene el minimo del histograma.
    '''
    counts, edges = np.histogram(array, bins=bins)
    mask = (edges[:-1] >= valleyRange[0]) & (edges[1:] <= valleyRange[1])
    validIdxs = np.where(mask)[0]
    cutIdx = validIdxs[np.argmin(counts[validIdxs])]
    cutValue = (edges[cutIdx] + edges[cutIdx + 1]) / 2
    return cutValue
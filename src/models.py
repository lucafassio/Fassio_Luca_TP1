import numpy as np
from tqdm import tqdm
from metrics import mse, rmse, mae, r2Score

class LinearRegresion:
    def __init__(self, df, labels, addBias=True):
        '''
        Clase principal para los modelos
            Args:
                df (pd.DataFrame): Informacion de entrenamiento sin los labels.
                labels (pd.DataFrame): Vector de targets de entrenamiento paralelos al dataset.
                addBias (bool): Determina si vamos a agregar un bias a nuestro modelo.
        '''
        self.addBias = addBias

        self.features = list(df.columns)
        self.data = df.to_numpy(dtype=float)
        self.target = labels.to_numpy(dtype=float).reshape(-1)

        if self.addBias:
            self.features = ['bias'] + self.features
            biasArr = np.ones((self.data.shape[0], 1), dtype=float)
            self.data = np.hstack((biasArr, self.data))

        self.coef = np.zeros(len(self.features), dtype=float)
        self.trained = False
        self.convergenceHistorial = []

    def _innerPred(self, X):
        return np.dot(X, self.coef)
    
    def _resetWeights(self):
        self.trained = False
        self.convergenceHistorial = []
        self.coef = np.zeros(len(self.features), dtype=float)


    def prediction(self, df):
        '''
        Predice por filas de un dataset dado.
            Args:
                df (pd.DataFrame): Nuevos datos, sin target.

            Returns:
                np.array: Predicciones del modelo.
        '''
        X = df.to_numpy(dtype=float)

        # Si el modelo usa bias, se agrega tambien en prediccion
        if self.addBias:
            biasArr = np.ones((X.shape[0], 1), dtype=float)
            X = np.hstack((biasArr, X))

        return self._innerPred(X)
        

    def fitByInverse(self):
        '''
        Entrena el modelo con la data interna que tiene.
            Usa el metodo de la inversa.
        '''
        self._resetWeights()

        X = self.data
        y = self.target

        self.coef = np.linalg.pinv(X.T @ X) @ X.T @ y
        self.trained = True


    def fitGradientDescent(self, rate=0.01, maxIters=1000, tol=1e-8):
        '''
        Entrena el modelo con la data interna que tiene.
            Usa el metodo de Gradient Descent

            Args:
                rate (float): Learning rate.
                maxIters (int): Cantidad maxima de iteraciones.
                tol (float): Margen de corte para la convergencia del metodo.
        '''
        self._resetWeights()

        X = self.data
        y_real = self.target
        n = X.shape[0]
        lastCost = float('inf')

        iterator = tqdm(range(maxIters), desc='Entrenando con Gradient Descent')
        for _ in iterator:
            
            pred = self._innerPred(X)
            err = pred - y_real
            gradient = (2/n) * (np.dot(X.T, err))

            self.coef = self.coef - rate * gradient

            cost = np.mean(err ** 2)
            self.convergenceHistorial.append(cost)
            iterator.set_postfix({'cost': f'{cost:.6f}'})

            if np.abs(lastCost - cost) < tol:
                self.trained = True
                return
            
            lastCost = cost
            
        print('El metodo no convergio, se alcanzo la maxima cantidad de iteraciones.\nEl modelo no se termino de entrenar correctamente.')

    def evaluate(self, df, labels, desnorm=None):
        '''
        Evalua el modelo con un dataset dado e imprime metricas.
        '''
        if not self.trained:
            print('Flaco, entrena el modelo.')
            return

        pred = self.prediction(df)
        y_real = labels.to_numpy(dtype=float).reshape(-1)

        if desnorm:
            pred = desnorm(pred)
            y_real = desnorm(y_real)

        mse_val = mse(y_real, pred)
        rmse_val = rmse(y_real, pred)
        mae_val = mae(y_real, pred)
        r2_val = r2Score(y_real, pred)

        print(f'MSE  : {mse_val:.6f}')
        print(f'RMSE : {rmse_val:.6f}')
        print(f'MAE  : {mae_val:.6f}')
        print(f'R2   : {r2_val:.6f}')

        return {
            'mse': mse_val,
            'rmse': rmse_val,
            'mae': mae_val,
            'r2': r2_val
        }
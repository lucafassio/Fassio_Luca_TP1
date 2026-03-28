import numpy as np
from metrics import mse, rmse, mae, r2Score
from data_splitting import splitData
from preprocessing import normalizeData

class LinearRegression:
    def __init__(self, df, labels, addBias=False, l2Penalty=0.0, l1Penalty=0.0):
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
        self.l2Penalty = l2Penalty
        self.l1Penalty = l1Penalty


    # Metodos internos

    def _innerPred(self, X):
        return np.dot(X, self.coef)
    

    def _resetWeights(self):
        self.trained = False
        self.convergenceHistorial = []
        self.coef = np.zeros(len(self.features), dtype=float)


    def _mseCost(self, X, y_real, coef=None):
        w = self.coef if coef is None else coef
        pred = np.dot(X, w)
        err = pred - y_real
        return np.mean(err ** 2)


    def _l1PenaltyValue(self, coef=None):
        w = self.coef if coef is None else coef
        if self.addBias:
            return np.sum(np.abs(w[1:]))
        return np.sum(np.abs(w))


    def _lassoObjective(self, X, y_real, lam, coef=None):
        return self._mseCost(X, y_real, coef=coef) + lam * self._l1PenaltyValue(coef=coef)


    def _mseGradient(self, X, y_real, coef=None):
        w = self.coef if coef is None else coef
        n = X.shape[0]
        pred = np.dot(X, w)
        err = pred - y_real
        return (2 / n) * np.dot(X.T, err)


    def _softThreshold(self, coef, threshold):
        out = coef.copy()
        startIdx = 1 if self.addBias else 0
        out[startIdx:] = np.sign(out[startIdx:]) * np.maximum(np.abs(out[startIdx:]) - threshold, 0.0)
        return out


    def _istaCandidate(self, X, y_real, lam, alpha):
        grad = self._mseGradient(X, y_real)
        gradientStep = self.coef - alpha * grad
        proxStep = self._softThreshold(gradientStep, alpha * lam)
        if self.addBias:
            proxStep[0] = gradientStep[0]
        return proxStep


    # Metodos publicos

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


    def fitGradientDescent(self, rate=0.01, maxIters=10000, tol=1e-8):
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

        for _ in range(maxIters):
            
            pred = self._innerPred(X)
            err = pred - y_real
            gradient = (2/n) * (np.dot(X.T, err))

            self.coef = self.coef - rate * gradient

            cost = np.mean(err ** 2)
            self.convergenceHistorial.append(cost)

            if np.abs(lastCost - cost) < tol:
                self.trained = True
                return
            
            lastCost = cost
            
        print('El metodo no convergio, se alcanzo la maxima cantidad de iteraciones.\nEl modelo no se termino de entrenar correctamente.')


    def fitRidgeByInverse(self, l2Penalty=None):
        '''Entrena el modelo con regularizacion L2 (ridge) usando la inversa.'''
        self._resetWeights()

        X = self.data
        y = self.target
        lam = self.l2Penalty if l2Penalty is None else float(l2Penalty)

        reg = np.eye(X.shape[1], dtype=float)
        if self.addBias:
            reg[0, 0] = 0.0

        self.coef = np.linalg.pinv(X.T @ X + lam * reg) @ X.T @ y
        self.trained = True


    def fitLassoGradientDescent(self, l1Penalty=None, rate=0.001, maxIters=30000, tol=1e-6):
        '''Entrena el modelo con regularizacion L1 (lasso) por gradient descent.'''
        self._resetWeights()

        X = self.data
        y_real = self.target
        lam = self.l1Penalty if l1Penalty is None else float(l1Penalty)
        lastCost = float('inf')

        for _ in range(maxIters):
            nextCoef = self._istaCandidate(X, y_real, lam, rate)

            self.coef = nextCoef

            cost = self._lassoObjective(X, y_real, lam)
            self.convergenceHistorial.append(cost)

            if np.abs(lastCost - cost) < tol:
                self.trained = True
                return

            lastCost = cost

        print('El metodo no convergio, se alcanzo la maxima cantidad de iteraciones.\nSe devuelven los mejores pesos encontrados hasta el momento.')
        self.trained = True


    def evaluate(self, df, labels, desnorm=None, printMetrics=True):
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

        if printMetrics == True:
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
   
    
    def printCoefficients(self):
        '''
        Imprime los coeficientes del modelo.
        '''
        if not self.trained:
            print('Flaco, entrena el modelo.')
            return

        print('Coeficientes del modelo:')
        for feature, coef in zip(self.features, self.coef):
            print(f'{feature}: {coef:.6f}')
    


def benitoAndaALaPlaya(features, df_ba):
    folds_ba = splitData(df_ba, trainPart=80, stratify='pileta', folds=5)
    truePoolPrices = np.full(df_ba.shape[0], np.nan, dtype=float)
    falsePoolPrices = np.full(df_ba.shape[0], np.nan, dtype=float)

    for train, valid in folds_ba:
        train, valid = normalizeData(train, valid)
        modelBenito = LinearRegression(train[features], train['precio'], addBias=True)
        modelBenito.fitByInverse()

        truePoolValid = valid[features].copy()
        falsePoolValid = valid[features].copy()
        
        truePoolValid.loc[:, 'pileta'] = 1
        falsePoolValid.loc[:, 'pileta'] = 0

        truePoolPred = modelBenito.prediction(truePoolValid)
        falsePoolPred = modelBenito.prediction(falsePoolValid)

        for j in range(len(valid)):
            idx = valid.index[j]
            pos = df_ba.index.get_loc(idx)

            truePoolPrices[pos] = truePoolPred[j]
            falsePoolPrices[pos] = falsePoolPred[j]

    return truePoolPrices, falsePoolPrices

import numpy as np
from IPython.display import clear_output
from models.SSA import SSA
from tqdm import tqdm

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

class LRF():
    def __init__(self, dim = None, max_dim = None, validation_size = 0.05):
        self._dim = dim
        self.max_dim = max_dim
        self.validation_size = validation_size

    def fit(self, X):
        '''
        X - series vector (1, n)
        '''
        validation_size = int(self.validation_size * X.shape[0])
        stop_index = X.shape[0] - validation_size

        initial_mse = np.inf
        if not self._dim:
            if self.max_dim < X.shape[0] // 2 + 10:
                max_range = self.max_dim
            else:
                max_range = X.shape[0] // 2 + 10
            for dim in tqdm(range(3,max_range)):
                self._dim = dim

                X_train = np.vstack([X[i:dim+i] for i in range(stop_index - dim )])
                y_train = np.stack([X[dim + i] for i in range(stop_index - dim )])

                y_validation = np.stack([X[stop_index + i] for i in range(validation_size)])

                regressor = LinearRegression(fit_intercept = False)
                regressor.fit(X_train, y_train)

                self.weights = regressor.coef_

                predictions = self.predict(X_train[-1, :], M = validation_size)
                mse_score = mean_squared_error(y_validation, predictions)
                if mse_score < initial_mse:
                    self.best_dim = dim
                    self.best_weights = self.weights
                    initial_mse = mse_score
            self._dim = self.best_dim
            self.weights = self.best_weights

            return self

        else:
            X_train = np.vstack([X[i:self._dim+i] for i in range(stop_index - self._dim )])
            y_train = np.stack([X[self._dim + i] for i in range(stop_index - self._dim )])
            regressor = LinearRegression(fit_intercept = False)
            regressor.fit(X_train, y_train)

            self.weights = regressor.coef_



    def predict(self, X, M):
        '''
        X - series vector (1, n), n >= dim
        M - number of points to predict
        '''
        for i in range(M):
            last_prediction = self.weights @ X[-self._dim:]
            X = np.concatenate([X, [last_prediction]])

        predictions = X[-M:]
        return predictions
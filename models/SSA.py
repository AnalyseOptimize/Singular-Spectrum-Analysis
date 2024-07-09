import numpy as np
from tqdm import tqdm
from math import floor


def henkelization(X):
    '''
    X: numpy 2d array
    
    return: numpy 1d array
    '''
    X_rev = X[::-1]
    return np.array([X_rev.diagonal(i).mean() for i in range(-X.shape[0]+1, X.shape[1])])


class SSA:
    def __init__(self, l = None, rank = None, chunk_size = None, intersection_rate=1):
        '''
        l: int, window size
        rank: int, restriction for rank in SVD
        chunk_size: int, size of each chunk. If None SSA is performed to the whole time series
        intersection_rate: float, intersection_rate * l observations will be in intersection (only for chunk mode)
        '''
        self.l = l if l is not None else 10
        self.rank = rank
        self.chunk_size = chunk_size
        self.intersection_rate = intersection_rate
        if chunk_size is not None:
            assert intersection_rate * l < chunk_size, "Invalid value, try to set parameters to fit the condition: intersection_rate * l < chunk_size"
            assert l < chunk_size, "Invalid value, try to set parameters to fit the condition: l < chunk_size"
        
        
        
    def _transform_to_matrix(self, series):
        '''
        series: numpy 1d array object with length n
    
        return: np.array 3-d tensor shape (rank, l, n-l+1)
        '''
        K = series.shape[0] - self.l + 1
        X = np.column_stack([series[i:self.l+i] for i in range(K)])


        U, sigma, VT = np.linalg.svd(X, full_matrices = False, hermitian = False)

        X_output = []

        if self.rank != None and self.rank < len(sigma):
            bound = self.rank
        else: bound = len(sigma)

        for i in range(bound):

            if sigma[i] > 0:
                X_i = sigma[i] * U[:,i].reshape(U.shape[0], 1) @ VT[i,:].reshape(1, VT.shape[1])
                X_output.append(X_i)

        del U, VT

        return np.array(X_output), sigma

    
    def _transform_to_series(self, series):
        '''
        series: numpy 1d array object with length

        return: np.array 2-d tensor shape (rank, n)
        '''
        K = series.shape[0] - self.l + 1
        X = np.column_stack([series[i:self.l+i] for i in range(K)])


        U, sigma, VT = np.linalg.svd(X, full_matrices = False, hermitian = False)

        X_output = []

        if self.rank != None and self.rank < len(sigma):
            bound = self.rank
        else:
            bound = len(sigma)
            
        for i in range(bound):
            if sigma[i] > 0:
                X_i = sigma[i] * U[:,i].reshape(U.shape[0], 1) @ VT[i,:].reshape(1, VT.shape[1])
                X_output.append(X_i)
                
        X_output = [henkelization(X_i) for X_i in X_output]

        del U, VT

        return np.array(X_output), sigma
    
    def transform_to_matrix(self, series):
        if self.chunk_size is None:
            return self._transform_to_matrix(series)
        else: NotImplementedError
        
        # self.rank = self.rank if self.rank < self.l else self.l
        
        # X_output, sigma = np.zeros((self.l,len(series) - self.l + 1)), np.zeros(self.rank)
        # intersection_times = np.zeros(len(series) - self.l + 1)
        # start_index, end_index = 0, self.chunk_size
        # intersection = self.intersection_rate * l
        # n_chunks = floor( (len(series) - self.chunk_size + 1) / (self.chunk_size - intersection))
        
        # for _ in tqdm(range(n_chunks),total = n_chunks, desc = 'Processing chunks'):
        #     X_chunk, sigma_chunk = self._transform_to_matrix(series[start_index:end_index])
        #     X_output[:, start_index:end_index] += X_chunk
        #     sigma += sigma_chunk

        #     start_index += self.chunk_size - intersection
        #     X_output[:, start_index:end_index]
        #     end_index += self.chunk_size - intersection

        # sigma /= n_chunks

    def transform_to_series(self, series):
        if self.chunk_size is None:
            return self._transform_to_series(series)
        
        self.rank = self.rank if self.rank < self.l else self.l

        intersection_times = np.zeros(len(series))
        tranformed_series = np.zeros((self.rank, len(series)))
        start_index, end_index = 0, self.chunk_size
        intersection = floor(self.intersection_rate * self.l)
        n_chunks = floor((len(series) - self.chunk_size) / (self.chunk_size - intersection)) + 1
        
        for iteration in tqdm(range(n_chunks+1), total = n_chunks+1, desc = "Processing chunks"):

            chunk, _ = self._transform_to_series(series[start_index:end_index])
            tranformed_series[:, start_index:end_index] += chunk

            start_index += self.chunk_size - intersection
            intersection_times[start_index:end_index] += 1
            end_index += self.chunk_size - intersection

        tranformed_series[:, intersection_times > 0] /= intersection_times[intersection_times > 0]
        return tranformed_series
        


    def fit(self, series):
        '''
        series: numpy 1d array object with length n
    
        return: self
        '''
        self._last_points = series[-self.l+1:]

        K = series.shape[0] - self.l + 1
        X = np.column_stack([series[i:self.l+i] for i in range(K)])


        U, sigma, VT = np.linalg.svd(X, full_matrices = False, hermitian = False)

        if self.rank != None and self.rank < len(sigma):
            bound = self.rank
        else:
            bound = len(sigma)

        self._U, self._VT, self._sigma = U[:, :bound], VT[:bound, :], sigma[:bound]

        del U, VT, sigma

        return self

    def predict(self, horizon):
        '''
        horizon: int - how many steps ahead to predict

        returns: numpy.1darray (1, M) - predictions
        '''
        Z = self._U[:self.l-1, :]

        last_points = self._last_points.copy()
        predictions = []
        
        for i in range(horizon):
            q = last_points.reshape(self.l-1, 1)
            h = np.linalg.inv(Z.T@Z) @ Z.T @ q # (r,1)
            
            prediction = self._U[-1, :] @ h
            predictions.append(prediction[0])

            last_points = np.roll(last_points, -1)
            last_points[-1] = prediction

        return np.array(predictions)









def w_corr(series, l):
    '''
    series: np.2darray of size (num_series, time_series_dimension)
    l: int - window length

    returns
    np.2darray - weighted correlation matrix
    '''
    
    weights = np.zeros(series.shape[1])
    # 1,2,3...min(L, N-L+1) for first values
    weights[ : min([l, series.shape[1] - l + 1])] += np.arange(1, min([l, series.shape[1] - l + 1])+1)
    # min(L, N-L+1) for median values
    weights[min([l, series.shape[1] - l + 1]) : max([l, series.shape[1] - l + 1])] += min([l, series.shape[1] - l + 1])
    # N - max(L, N-L+1), ... 1 for last values
    weights[max([l, series.shape[1] - l + 1]) : ] += series.shape[1] - np.arange(max([l, series.shape[1] - l + 1]), series.shape[1])
    weights /= weights.sum()

    series = (series - series.mean(1).reshape(series.shape[0], 1)) / (series.std(1) + 10**-6).reshape(series.shape[0], 1) 
    return (series * weights) @ series.T

            
            
import numpy as np
from .metrics import euclidean

@np.vectorize
def max_(a, b):
    return max(a, b)

class LOF():
    def __init__(self, k, metric=euclidean):
        self.metric = metric
        self.k = k
    
    def nearest_neighbors(self, data_points):
        distances = []
        for A in data_points:
            d = []
            for B in data_points:
                d.append(self.metric(A, B))
            distances.append(d)
        
        distances = np.array(distances)
        nn = np.argsort(distances, axis=1)[:, 1:(self.k + 1)]

        return nn, np.sort(distances, axis=1)[:, 1:(self.k + 1)]
    
    def _lrd(self, data_points, x):
        data_points = np.concatenate((data_points, [x]))
        nn, nn_dist = self.nearest_neighbors(data_points)
        x_nn_dist = nn_dist[-1, :]
        x_nn = nn[-1, :]
        k_dist = nn_dist[x_nn, -1]
        reachability = np.sum(max_(x_nn_dist, k_dist))
        lrd = self.k / reachability
        return lrd
    
    def _lof(self, data_points, x):
        data_points = np.concatenate((data_points, [x]))
        lrd_x = self._lrd(data_points, x)
        nn, nn_dist = self.nearest_neighbors(data_points)
        x_nn = nn[-1, :]
        lrd_nn_sum = 0
        for idx_nn in x_nn:
            lrd_nn_sum += self._lrd(data_points, data_points[idx_nn])
        
        return lrd_nn_sum / (lrd_x * self.k)
    
    def fit(self, X):
        self.data_points = X
    
    def _fit_predict(self, X):
        for i in range(len(X)):
            self.fit(np.concatenate((X[:i], X[i+1:])))
            yield self.lof(X[i])

    def fit_predict(self, X):
        return np.array(list(self._fit_predict(X)))
    
    def lof(self, x):
        return self._lof(self.data_points, x)
    
    def _predict(self, X):
        for x in X:
            yield self.lof(x)
    
    def predict(self, X):
        return np.array(list(self._predict(X)))

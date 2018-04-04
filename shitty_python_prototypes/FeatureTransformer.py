import numpy as np

class FeatureTransformer:
    def __init__(self, n_bins):
        self.n = n_bins / 2
        
    def get_percentile_threshold(self, feature):
        perc_q = np.arange(0, 100, 100.0 / (self.n + 1))[1:]
        percentiles = []
        for q in perc_q:
            percentiles.append(np.percentile(feature, q))
        return np.array(percentiles)

    def get_uniform_threshold(self, feature):
        max_val = np.max(feature)
        min_val = np.min(feature)
        return np.arange(min_val, max_val, (max_val - min_val) / (self.n + 1))[1:]

    def get_threshold(self, feature):
        p = self.get_percentile_threshold(feature)
        u = self.get_uniform_threshold(feature)
        t = np.hstack([[-np.inf], p, u, [np.inf]])
        return sorted(t)

    def fit(self, X):
        l, m = X.shape  
        self.tresholds = []
        for j in range(m):
            t = self.get_threshold(X[:, j])
            self.tresholds.append(t)

    def transform(self, X): 
        X_new = np.zeros_like(X)
        l, m = X.shape   
        for i in range(l):
            for j in range(m):
                for k in range(2 * self.n + 1):
                    if X[i, j] > self.tresholds[j][k] and X[i, j] <= self.tresholds[j][k + 1]:
                        X_new[i, j] = k
        
        return X_new

    def get_sizes(self):
        return len(self.tresholds), len(self.tresholds[0]) - 1

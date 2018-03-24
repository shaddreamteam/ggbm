import numpy as np

def get_percentile_threshold(feature, n):
    perc_q = np.arange(0, 100, 100.0 / (n + 1))[1:]
    percentiles = []
    for q in perc_q:
        percentiles.append(np.percentile(feature, q))
    return np.array(percentiles)

def get_uniform_threshold(feature, n):
    max_val = np.max(feature)
    min_val = np.min(feature)
    return np.arange(min_val, max_val, (max_val - min_val) / (n + 1))[1:]

def get_threshold(feature, n):
    p = get_percentile_threshold(feature, n)
    u = get_uniform_threshold(feature, n)
    t = np.hstack([[min(feature) - 1e-3], p, u, [max(feature) + 1e-3]])
    return sorted(t)

def transform_features(X, n):
    l, m = X.shape
    X_new = np.zeros(shape=(l, m))
    tresholds = []
    for j in range(m):
        t = get_threshold(X[:, j], n)
        tresholds.append(t)
        for i in range(l):
            for k in range(2 * n + 1):
                if X[i, j] > t[k] and X[i, j] <= t[k + 1]:
                    X_new[i, j] = k
                
    return tresholds, X_new


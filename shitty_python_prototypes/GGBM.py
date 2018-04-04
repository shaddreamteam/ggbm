import numpy as np
from Tree import Tree
from FeatureTransformer import FeatureTransformer
from dataset_transform import OptEntry


class GGBM:
    def __init__(self, depth, loss, n_bins, lambda_val, gamma, eps, n_estimators):
        self.depth = depth
        self.loss = loss
        self.n_bins = n_bins
        self.lambda_val = lambda_val
        self.gamma = gamma
        self.eps = eps
        self.n_estimators = n_estimators
        self.transformer = FeatureTransformer(self.n_bins)

    def get_first_entries(self, X, y):
        self.transformer.fit(X)
        base_prediction = np.ones_like(y) * self.base
        x = self.transformer.transform(X)
        g, h = self.loss(y, base_prediction)
        self.entries = []
        for x_, y_, g_, h_ in zip(x, y, g, h):
            self.entries.append(OptEntry(x_, y_, g_, h_, self.base))
    
    def fit(self, X, y):
        self.base = np.mean(y)
        self.get_first_entries(X, y)
        self.trees = []
        n_features, n_bins = self.transformer.get_sizes()
        for _ in range(self.n_estimators):
            tree = Tree(self.depth, self.loss, self.entries, n_features, n_bins, 
                        self.lambda_val, self.gamma)
            tree.construct()
            self.trees.append(tree)
            if tree.real_depth < 1:
                break
            for i in range(len(self.entries)):
                prediction = tree.predict(self.entries[i].x)
                self.entries[i].prediction += self.eps * prediction
                self.entries[i].g, self.entries[i].h = \
                    self.loss(self.entries[i].y, self.entries[i].prediction)
                    
    def predict(self, X):
        X_new = self.transformer.transform(X)
        predictions = []
        for x in X_new:
            prediction = self.base
            for tree in self.trees:
                prediction += self.eps * tree.predict(x)
            predictions.append(prediction)
        return np.array(predictions)

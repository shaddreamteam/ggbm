class MSE:
    def __init__(self, n):
        self.n = n

    def __call__(self, y_true, y_predicted):
        grad = 2.0 / self.n * (y_predicted - y_true)
        hess = 2.0 / self.n * np.ones_like(y_predicted)
        return grad, hess

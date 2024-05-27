import numpy as np

class LinearNeuron():
    def __init__(self, X, lr=0.1, n_iter=100):
        # Initialisation de W, b
        self.W = np.random.randn(X.shape[1], 1)
        self.b = np.random.randn(1)
        self.lr = lr
        self.n_iter = n_iter

    def log_loss(self, A, y):
        #Calcule du coup pour une itération
        return 1 / len(y) * np.sum(-y * np.log(A) - (1 - y) * np.log(1 - A))

    def apply_sigmoide(self, X):
        Z = X.dot(self.W) + self.b
        A = 1 / (1 + np.exp(-Z))
        return A

    def gradients(self, A, X, y):
        dW = 1 / len(y) * np.dot(X.T, A - y)
        db = 1 / len(y) * np.sum(A - y)
        return (dW, db)
    
    def update(self, dW, db):
        self.W = self.W - self.lr * dW
        self.b = self.b - self.lr * db

    def fit(self, X, y):
        Loss = []

        for i in range(self.n_iter):
            A = self.apply_sigmoide(X)
            Loss.append(self.log_loss(A, y))
            dW, db = self.gradients(A, X, y)
            self.update(dW, db)

        return Loss

    def predict(self, X):
        A = self.apply_sigmoide(X)
        return A >= 0.5
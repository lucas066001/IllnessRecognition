import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_circles
from sklearn.metrics import accuracy_score, log_loss
from tqdm import tqdm

class NeuralNetwork():
    def __init__(self, X, y, n1, lr=0.1, n_iter=100):
        # Initialisation de W1, b1, ...
        n0 = X.shape[0]
        n2 = y.shape[0]
        
        W1 = np.random.randn(n1, n0)
        b1 = np.zeros((n1, 1))
        W2 = np.random.randn(n2, n1)
        b2 = np.zeros((n2, 1))

        self.parameters = {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2
        }
        self.lr = lr
        self.n_iter = n_iter

    def forward_propagation(self, X):

        W1 = self.parameters['W1']
        b1 = self.parameters['b1']
        W2 = self.parameters['W2']
        b2 = self.parameters['b2']

        Z1 = W1.dot(X) + b1
        A1 = 1 / (1 + np.exp(-Z1))

        Z2 = W2.dot(A1) + b2
        A2 = 1 / (1 + np.exp(-Z2))

        activations = {
            'A1': A1,
            'A2': A2
        }

        return activations
    
    def back_propagation(self, X, y, activations):

        A1 = activations['A1']
        A2 = activations['A2']
        W2 = self.parameters['W2']

        m = y.shape[1]

        dZ2 = A2 - y
        dW2 = 1 / m * dZ2.dot(A1.T)
        db2 = 1 / m * np.sum(dZ2, axis=1, keepdims = True)

        dZ1 = np.dot(W2.T, dZ2) * A1 * (1 - A1)
        dW1 = 1 / m * dZ1.dot(X.T)
        db1 = 1 / m * np.sum(dZ1, axis=1, keepdims = True)

        gradients = {
            'dW1' : dW1,
            'db1' : db1,
            'dW2' : dW2,
            'db2' : db2
        }
        
        return gradients


    def update(self, gradients):

        W1 = self.parameters['W1']
        b1 = self.parameters['b1']
        W2 = self.parameters['W2']
        b2 = self.parameters['b2']

        dW1 = gradients['dW1']
        db1 = gradients['db1']
        dW2 = gradients['dW2']
        db2 = gradients['db2']

        W1 = W1 - self.lr * dW1
        b1 = b1 - self.lr * db1
        W2 = W2 - self.lr * dW2
        b2 = b2 - self.lr * db2

        self.parameters = {
            'W1': W1,
            'b1': b1,
            'W2': W2,
            'b2': b2
        }
        return

    # def log_loss(self, A, y):
    #     #Calcule du cout pour une itération
    #     epsilon = 1e-15
    #     return 1 / len(y) * np.sum(-y * np.log(A + epsilon) - (1 - y) * np.log(1 - A + epsilon))

    def predict(self, X):
        activations = self.forward_propagation(X)
        A2 = activations['A2']
        return A2 >= 0.5
    
    def fit(self, X, y):
        # initialisation parameters
        train_loss = []
        train_acc = []
        history = []

        # gradient descent
        for i in tqdm(range(self.n_iter)):
            activations = self.forward_propagation(X)
            A2 = activations['A2']

            # Sauvegarde historique des valeurs
            train_loss.append(log_loss(y.flatten(), A2.flatten()))
            y_pred = self.predict(X)
            train_acc.append(accuracy_score(y.flatten(), y_pred.flatten()))
            history.append([self.parameters.copy(), train_loss, train_acc, i])
            if i %100 == 0:
                # Plot courbe d'apprentissage
                plt.figure(figsize=(12, 4))
                plt.subplot(1, 2, 1)
                plt.plot(train_loss, label='train loss')
                plt.legend()
                plt.subplot(1, 2, 2)
                plt.plot(train_acc, label='train acc')
                plt.legend()
                plt.show()

            # mise a jour
            gradients = self.back_propagation(X, y, activations)
            self.update(gradients)

        return history
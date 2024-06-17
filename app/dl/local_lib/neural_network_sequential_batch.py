import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_circles
from sklearn.metrics import accuracy_score, log_loss, recall_score, precision_score
from tqdm import tqdm
import math

class NeuralNetworkMultiLayerSequentialStrat():
    def __init__(self, X, y, dimensions = (16, 16, 16), lr=0.1, n_iter=1000, test_size=0.2, strategy="full", sub_parts=5):
        self.lr = lr
        self.n_iter = n_iter
        self.test_size = test_size
        self.dimensions = list(dimensions)
        self.dimensions.insert(0, X.shape[0])
        self.dimensions.append(y.shape[0])
        self.parameters = {}
        self.strategy = strategy
        self.sub_parts = sub_parts

        # Initialisation de W1, b1, ...
        C = len(self.dimensions)
        for c in range(1, C):
            self.parameters['W' + str(c)] = np.random.randn(self.dimensions[c], self.dimensions[c - 1])
            self.parameters['b' + str(c)] = np.random.randn(self.dimensions[c], 1)


    def forward_propagation(self, X):
        activations = {'A0': X}

        C = len(self.parameters) // 2

        for c in range(1, C + 1):
            if(not np.any(activations['A' + str(c - 1)])):
                print("Nan in activations")
                raise ValueError("Nan found")
            Z = self.parameters['W' + str(c)].dot(activations['A' + str(c - 1)]) + self.parameters['b' + str(c)]
            activations['A' + str(c)] = 1 / (1 + np.exp(-Z))

        return activations
    
    def back_propagation(self, y, activations):

        m = y.shape[1]
        C = len(self.parameters) // 2
        dZ = activations['A' + str(C)] - y
        gradients = {}

        for c in reversed(range(1, C + 1)):
            gradients['dW' + str(c)] = 1/m * np.dot(dZ, activations['A' + str(c - 1)].T)
            gradients['db' + str(c)] = 1/m * np.sum(dZ, axis=1, keepdims=True)
            #N'a pas de sens pour Z0
            if c > 1:
                dZ = np.dot(self.parameters['W' + str(c)].T, dZ) * activations['A' + str(c - 1)] * (1 - activations['A' + str(c - 1)])

        return gradients

    def update(self, gradients):

        C = len(self.parameters) // 2

        for c in range(1, C + 1):

            self.parameters['W' + str(c)] = self.parameters['W' + str(c)] - self.lr * gradients['dW' + str(c)]
            self.parameters['b' + str(c)] = self.parameters['b' + str(c)] - self.lr * gradients['db' + str(c)]

        return

    def predict(self, X):
        activations = self.forward_propagation(X)
        C = len(self.parameters) // 2
        Af = activations['A' + str(C)]
        return Af >= 0.5
    
    def fit(self, X_train, X_test, y_train, y_test):

        training_history = np.zeros((int(self.n_iter) + 1, 3))

        C = len(self.parameters) // 2

        # gradient descent
        if(self.strategy == "full"):
            for i in tqdm(range(self.n_iter)):
                activations = self.forward_propagation(X_train)
                gradients = self.back_propagation(y_train, activations)
                self.update(gradients)
                Af = activations['A' + str(C)]
                # calcul du log_loss et de l'accuracy
                training_history[i, 0] = (log_loss(y_train.flatten(), Af.flatten()))
                y_pred_train = self.predict(X_train)
                y_pred_test = self.predict(X_test)
                training_history[i, 1] = (accuracy_score(y_train.flatten(), y_pred_train.flatten()))
                training_history[i, 2] = (accuracy_score(y_test.flatten(), y_pred_test.flatten()))

        elif(self.strategy == "sub"):
            
            self.n_iter = (np.floor(self.n_iter / self.sub_parts)).astype(int)
            portion = 1 / self.sub_parts

            nb_element_sub_train = math.floor(X_train.shape[1] * portion)
            nb_element_sub_test = math.floor(X_test.shape[1] * portion)
            e = 0
            for x in range(0, self.sub_parts):
                start_train_index = nb_element_sub_train * x
                end_train_index = nb_element_sub_train * (x+1)
                start_test_index = nb_element_sub_test * x
                end_test_index = nb_element_sub_test * (x+1)
                X_train_sub = X_train[:, start_train_index:end_train_index]
                X_test_sub = X_test[:, start_test_index:end_test_index]
                y_train_sub = y_train[:, start_train_index:end_train_index]
                y_test_sub = y_test[:, start_test_index:end_test_index]
                print(X_train_sub.shape)
                print(X_test_sub.shape)
                
                for i in tqdm(range(self.n_iter)):
                    e+=1
                    activations = self.forward_propagation(X_train_sub)
                    gradients = self.back_propagation(y_train_sub, activations)
                    self.update(gradients)
                    Af = activations['A' + str(C)]
                    # calcul du log_loss et de l'accuracy
                    training_history[e, 0] = log_loss(y_train_sub.flatten(), Af.flatten())
                    y_pred_train = self.predict(X_train_sub)
                    y_pred_test = self.predict(X_test_sub)
                    training_history[e, 1] = accuracy_score(y_train_sub.flatten(), y_pred_train.flatten())
                    training_history[e, 2] = accuracy_score(y_test_sub.flatten(), y_pred_test.flatten())

        else:
            raise ValueError("Unsupported strategy")
        return training_history
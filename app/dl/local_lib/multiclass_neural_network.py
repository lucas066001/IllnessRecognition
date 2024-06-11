import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_circles
from sklearn.metrics import accuracy_score, log_loss
from tqdm import tqdm

class MultiClassNeuralNetwork():
    def __init__(self, X, y, dimensions = (16, 16, 16), lr=0.1, n_iter=1000, test_size=0.2):
        self.lr = lr
        self.n_iter = n_iter
        self.test_size = test_size
        self.dimensions = list(dimensions)
        self.dimensions.insert(0, X.shape[0])
        self.dimensions.append(y.shape[0])
        self.parameters = {}

        self.classes = np.unique(y)
        self.output_act = []
        self.divider  = len(np.unique(y)) + 1
        bornes = np.linspace(0,1, self.divider)
        self.output_act = []
        self.mapped_classes = {}

        for i in range(0, len(self.classes)):
            self.output_act.append([round(bornes[i], 2), round(bornes[i + 1], 2), self.classes[i]])
            self.mapped_classes[i] = round((round(bornes[i], 2) + round(bornes[i + 1], 2)) / 2, 2)
        print(self.mapped_classes)
        print(self.output_act)

        # Initialisation de W1, b1, ...
        C = len(self.dimensions)
        for c in range(1, C):
            self.parameters['W' + str(c)] = np.random.randn(self.dimensions[c], self.dimensions[c - 1])
            self.parameters['b' + str(c)] = np.random.randn(self.dimensions[c], 1)

    def get_corresponding_classes(self, acts):
        results = []


        for act in acts[0]:
            result = 0
            for i in range(len(self.output_act)):
                if self.output_act[i][0] <= act < self.output_act[i][1]:
                    result = self.output_act[i][2]
                    break  # On peut arrêter la boucle une fois que la classe correspondante est trouvée
            results.append(result)
        return np.array(results).reshape(1, -1)

    def forward_propagation(self, X):
        activations = {'A0': X}

        C = len(self.parameters) // 2

        for c in range(1, C + 1):

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
        return self.get_corresponding_classes(Af)
    
    def log_loss(self, A, y):
        #Calcule du cout pour une itération
        epsilon = 1e-15
        return 1 / len(y) * np.sum(-y * np.log(A + epsilon) - (1 - y) * np.log(1 - A + epsilon))

    def fit(self, X_train, X_test, y_train, y_test):

        training_history = np.zeros((int(self.n_iter), 3))

        C = len(self.parameters) // 2
        y_train_adjusted = np.array([self.mapped_classes[val] for val in y_train[0]]).reshape(1, -1)
        y_test_adjusted = np.array([self.mapped_classes[val] for val in y_test[0]]).reshape(1, -1)

        # gradient descent
        for i in tqdm(range(self.n_iter)):

            activations = self.forward_propagation(X_train)
            gradients = self.back_propagation(y_train_adjusted, activations)
            self.update(gradients)
            Af = activations['A' + str(C)]
            # calcul du log_loss et de l'accuracy
            y_pred_train = self.predict(X_train)
            y_pred_test = self.predict(X_test)
            # training_history[i, 0] = (log_loss(y_train.flatten(), y_pred_train.flatten(), labels=self.classes))
            training_history[i, 0] = (self.log_loss(Af.flatten(), y_train_adjusted.flatten()))
            # training_history[i, 1] = (accuracy_score(y_train_adjusted.flatten(), y_pred_train.flatten()))
            # training_history[i, 2] = (accuracy_score(y_test_adjusted.flatten(), y_pred_test.flatten()))

        return training_history
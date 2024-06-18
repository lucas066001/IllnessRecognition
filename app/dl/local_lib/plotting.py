from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, recall_score, precision_score
import matplotlib.pyplot as plt
import numpy as np

def plotHist(hist, y_true, y_pred, bar_width=0.2, zoomx=(1000, 1800), zoomy=(0.60,0.90)):

    ac = accuracy_score(np.argmax(y_true, axis=0),  np.argmax(y_pred, axis=0))
    re = recall_score(np.argmax(y_true, axis=0),  np.argmax(y_pred, axis=0), average='weighted', zero_division=1)
    pr = precision_score(np.argmax(y_true, axis=0),  np.argmax(y_pred, axis=0), average='weighted', zero_division=1)

    print("Least accuracy :" + str(ac))
    print("Least recall :" + str(re))
    print("Least precision :" + str(pr))
    
    plt.figure(figsize=(14, 6))
    plt.title('Model scores')
    plt.grid(True)
    plt.bar(1, ac, width=bar_width, label='Accuracy Score', color='#D1FF38')
    plt.bar(2, re, width=bar_width, label='Recall Score', color='#ACD32A')
    plt.bar(3, pr, width=bar_width, label='Precision Score', color='#84A21F')

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(hist[:, 0])
    plt.title('Loss Curve')
    plt.xlabel('Iterations')
    plt.ylabel('L')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(hist[:, 1].shape[0]), hist[:, 1], color=('green', 0.5), label='Training accuracy')
    plt.plot(range(hist[:, 2].shape[0]), hist[:, 2], color=('orange', 0.5),  label='Test accuracy')
    plt.title('Learning Curve')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()


    ax = plt.subplot(1, 1, 1)
    plt.plot(range(hist[:, 1].shape[0]), hist[:, 1], color=('green', 0.5), label='Training accuracy')
    plt.plot(range(hist[:, 2].shape[0]), hist[:, 2], color=('orange', 0.5), label='Test accuracy')
    ax.set_ylim(zoomy)
    ax.set_xlim(zoomx)
    plt.title('Learning Curve (zoomed)')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

    disp = ConfusionMatrixDisplay.from_predictions(np.argmax(y_true, axis=0), np.argmax(y_pred, axis=0))
    disp.figure_.suptitle("Confusion Matrix")
    plt.show()
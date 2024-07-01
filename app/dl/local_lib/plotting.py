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

def plotClassesAlong2PCa(pc_1, pc_2, y):
    plt.figure(figsize=(10, 6))

    plt.scatter(pc_1, pc_2, c=y, cmap=plt.cm.coolwarm, alpha=0.5)

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Decision Boundary')
    plt.show()

def plotClassesAlong3PCa(pc_1, pc_2, pc_3, y):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pc_1, pc_2, pc_3, c=y, cmap=plt.cm.coolwarm, alpha=0.3)

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Decision Boundary')
    plt.show()

def plotDecisionBoundary2D(mnn:any, X_test_pca_scaled, y_true, precision=0.1, overlap=0.2, true_op=0.4, boundary_op=0.1):
    pc_1 = X_test_pca_scaled[0, :]
    pc_2 = X_test_pca_scaled[1, :]
    etendu = np.abs(pc_1.max() - pc_1.min())

    # Créer une grille de points couvrant la plage de vos données
    xx, yy = np.meshgrid(np.arange(pc_1.min() - etendu * overlap,pc_1.max() + etendu * overlap,precision),
                        np.arange(pc_2.min() - etendu * overlap,pc_2.max() + etendu * overlap,precision))

    # Préparer les points de la grille pour la prédiction
    grid = np.c_[xx.ravel(), yy.ravel()]
    X_decision = np.zeros((50, grid.shape[0]))
    X_decision[:2, :] = grid.T
    preds = mnn.predict(X_decision)
    Z = np.argmax(preds, axis=0).reshape(xx.shape)

    # Tracer les points et la surface de décision
    plt.figure(figsize=(10, 6))
    plt.contourf(xx.T, yy.T, Z.T, alpha=boundary_op, cmap=plt.cm.coolwarm)
    plt.scatter(pc_1, pc_2, c=np.argmax(y_true, axis=0), cmap=plt.cm.coolwarm, alpha=true_op)

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Decision Boundary')
    plt.show()

def plotDecisionBoundary3D(mnn:any, X_test_pca_scaled, y_true, precision=0.1, overlap=0.2, true_op=0.4, boundary_op=0.1):
    pc_1 = X_test_pca_scaled[0, :]
    pc_2 = X_test_pca_scaled[1, :]
    pc_3 = X_test_pca_scaled[2, :]
    etendu_1 = np.abs(pc_1.max() - pc_1.min())
    etendu_2 = np.abs(pc_2.max() - pc_2.min())
    etendu_3 = np.abs(pc_3.max() - pc_3.min())

    # Créer une grille de points couvrant la plage de vos données
    xx, yy, zz = np.meshgrid(np.arange(pc_1.min() - etendu_1 * overlap,pc_1.max() + etendu_1 * overlap,precision),
                        np.arange(pc_2.min() - etendu_2 * overlap,pc_2.max() + etendu_2 * overlap,precision),
                        np.arange(pc_3.min() - etendu_3 * overlap,pc_3.max() + etendu_3 * overlap,precision))

    # Préparer les points de la grille pour la prédiction
    grid = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
    X_decision = np.zeros((50, grid.shape[0]))
    X_decision[:3, :] = grid.T
    preds = mnn.predict(X_decision)
    Z = np.argmax(preds, axis=0).reshape(xx.shape)
    # Tracer les points et la surface de décision
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(xx.T, yy.T, zz.T, c=Z.T, alpha=boundary_op, cmap=plt.cm.coolwarm)

    ax.scatter(pc_1, pc_2, pc_3, c=np.argmax(y_true, axis=0), cmap=plt.cm.coolwarm, alpha=true_op)

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Decision Boundary')
    plt.show()

def plotDecisionBoundaries(mnn:any, X_test_pca_scaled, y_true, precision=0.1, overlap=0.2, true_op=0.4, boundary_op=0.1):
    pc_1 = X_test_pca_scaled[0, :]
    pc_2 = X_test_pca_scaled[1, :]
    pc_3 = X_test_pca_scaled[2, :]
    etendu_1 = np.abs(pc_1.max() - pc_1.min())
    etendu_2 = np.abs(pc_2.max() - pc_2.min())
    etendu_3 = np.abs(pc_3.max() - pc_3.min())

    # Créer une grille de points couvrant la plage de vos données
    xx, yy, zz = np.meshgrid(np.arange(pc_1.min() - etendu_1 * overlap,pc_1.max() + etendu_1 * overlap,precision),
                    np.arange(pc_2.min() - etendu_2 * overlap,pc_2.max() + etendu_2 * overlap,precision),
                    np.arange(pc_3.min() - etendu_3 * overlap,pc_3.max() + etendu_3 * overlap,precision))

    # Préparer les points de la grille pour la prédiction
    grid = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
    X_decision = np.zeros((50, grid.shape[0]))
    X_decision[:3, :] = grid.T
    preds = mnn.predict(X_decision)
    Z = np.argmax(preds, axis=0).reshape(xx.shape)
    # Tracer les points et la surface de décision
    fig = plt.figure(figsize=(10, 6))
    ax = plt.subplot(1,2,1, projection='3d')

    ax.scatter(xx.T, yy.T, zz.T, c=Z.T, alpha=boundary_op, cmap=plt.cm.coolwarm)

    ax.scatter(pc_1, pc_2, pc_3, c=np.argmax(y_true, axis=0), cmap=plt.cm.coolwarm, alpha=true_op)

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Decision Boundary')

    xx, yy = np.meshgrid(np.arange(pc_1.min() - etendu_1 * overlap,pc_1.max() + etendu_1 * overlap,precision),
                    np.arange(pc_2.min() - etendu_2 * overlap,pc_2.max() + etendu_2 * overlap,precision))
    grid = np.c_[xx.ravel(), yy.ravel()]
    X_decision = np.zeros((50, grid.shape[0]))
    X_decision[:2, :] = grid.T
    preds = mnn.predict(X_decision)

    Z = np.argmax(preds, axis=0).reshape(xx.shape)
    print(preds.shape)

    # Tracer les points et la surface de décision
    fig = plt.figure(figsize=(16, 6))
    ax = plt.subplot(1,2,2)
    ax.contourf(xx.T, yy.T, Z.T, alpha=boundary_op, cmap=plt.cm.coolwarm)
    ax.scatter(pc_1, pc_2, c=np.argmax(y_true, axis=0), cmap=plt.cm.coolwarm, alpha=true_op)

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Decision Boundary')
    plt.show()
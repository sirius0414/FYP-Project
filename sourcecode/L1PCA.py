# L1-Norm Based Principal Component Analysis via Bit Flipping
import gc
import numpy as np
from time import time
from scipy.stats import loguniform
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from dataset import createDataset
import matplotlib.pyplot as plt
from l1pca_sbfk_v0 import l1pca_sbfk

def L1_PCA_trail(name,dir):
    t0 = time()
    myDataset = createDataset(dir)

    # Data Preparation
    X = myDataset.spectogram
    y1 = myDataset.label
    y2 = myDataset.grandelabel
    # shaping  X
    X = np.array(X)
    X_reshape = X.reshape(len(X), -1)
    y1 = np.array(y1)
    y2 = np.array(y2)

    # Splitting data
    X_train, X_test, y1_train, y1_test, y2_train, y2_test = train_test_split(X_reshape, y1, y2, test_size=0.2,
                                                                             random_state=42, stratify=y1)
    # size of train and test data
    print("Size of train data: %d" % len(X_train))
    print("Size of test data: %d" % len(X_test))

    # Standardization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print("Data ready time: %0.3fs" % (time() - t0))

    # L1-PCA with different number of components, from 5 to 50
    n_components = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    avg_acc = []
    for n in n_components:
        t0 = time()
        Q, B, vmax = l1pca_sbfk(X_train.T, n, 5, True)
        X_train_pcal1 = np.matmul(X_train, Q)
        X_test_pcal1 = np.matmul(X_test, Q)
        # SVM
        t_svc = time()
        param_grid = {
            "C": loguniform(1e3, 1e5),
            "gamma": loguniform(1e-4, 1e-1),
        }
        clf = RandomizedSearchCV(
            SVC(kernel="rbf", class_weight="balanced"), param_grid, n_iter=10
        )
        clf = clf.fit(X_train_pcal1, y2_train)
        print("-------------------------------------------------------")
        print(name)
        print(' L1 Norm PCA with %d components' % n)
        print('time for SVM: %0.3fs' % (time() - t_svc))
        print("Best estimator found by grid search:")
        print(clf.best_estimator_)
        # evaluation
        target_names = ["Drone", "Bird"]
        y_pred = clf.predict(X_test_pcal1)
        avg_acc.append(clf.score(X_test_pcal1, y2_test))
        print("Classification report for classifier %s:\n%s\n"
              % (clf, classification_report(y2_test, y_pred, target_names=target_names)))
        ConfusionMatrixDisplay.from_estimator(
            clf, X_test_pcal1, y2_test, display_labels=target_names, xticks_rotation="vertical"
        )
        # save the confusion matrix
        plt.tight_layout()
        plt.savefig('./outputs/L1PCA_with' + str(X.shape[2]) + 'bin' + 'CFM' + str(n) + '.png', dpi=300,
                    bbox_inches='tight')
        # Average accuracy
        print("PCA with %d components done in %0.3fs" % (n, time() - t0))
        print("-------------------------------------------------------")

    # Print Figure of Accuracy vs Number of Components
    plt.figure()
    # fixed y axis range 0-1:
    plt.ylim(0, 1)
    plt.plot(n_components, avg_acc, "b", label="Accuracy")
    plt.xlabel("Number of Components")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Number of Components")
    # Save the figure
    plt.savefig('./outputs/L1PCA_with' + str(X.shape[2]) + 'bin' + '.png', dpi=300, bbox_inches='tight')
    plt.show()
    # Delete Variables and Release Memory
    del X, X_reshape, X_train, X_test, y1, y2, y1_train, y1_test, y2_train, y2_test, scaler, Q, B, vmax
    del param_grid, clf, target_names, y_pred, n_components
    gc.collect()
    # return avg_acc
    return avg_acc

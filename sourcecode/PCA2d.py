### 2D PCA on dataset with SVM classifier
import gc
import numpy as np
import matplotlib.pyplot as plt
from time import time
from scipy.stats import loguniform
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.svm import SVC
from dataset import createDataset


def PCA_2D_trail(trail_name,dir):
    # path of dataset
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

    # 2D PCA with different number of components, from 5 to 60
    n_components = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    avg_acc = []
    X_train_reshape = X_train.reshape(len(X_train), X.shape[2], X.shape[1])
    X_test_reshape = X_test.reshape(len(X_test), X.shape[2], X.shape[1])
    # numbers of examples in train data
    n_train = X_train_reshape.shape[0]
    n_test = X_test_reshape.shape[0]
    for n in n_components:
        t0 = time()
        # average value of all images
        avg_train = np.mean(X_train_reshape, axis=0)
        # subtracting average value from all images
        X_train_centered = np.zeros((n_train, X_train_reshape.shape[1], X_train_reshape.shape[2]))
        X_test_centered = np.zeros((n_test, X_test_reshape.shape[1], X_test_reshape.shape[2]))
        for i in range(n_train):
            X_train_centered[i] = X_train_reshape[i] - avg_train
        for i in range(n_test):
            X_test_centered[i] = X_test_reshape[i] - avg_train
        # image covariance matrix
        cov_train = np.zeros((X.shape[2], X.shape[2]))
        for i in range(n_train):
            cov_train += np.dot(X_train_centered[i], X_train_centered[i].T)
        cov_train /= n_train
        # eigenvalues and eigenvectors of covariance matrix
        eigenvalues, eigenvectors = np.linalg.eig(cov_train)
        # sorting eigenvalues and eigenvectors
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        # choosing n eigenvectors with largest eigenvalues
        eigenvectors = eigenvectors[:, :n]
        # projection of train data on eigenvectors
        X_test_2dpca = np.zeros((n_test, X_test_centered.shape[2], n))
        X_train_2dpca = np.zeros((n_train, X_train_centered.shape[2], n))
        for i in range(n_train):
            X_train_2dpca[i] = np.dot(X_train_centered[i].T, eigenvectors)
        # reshape the 3D tensor of train and test data to 2D
        X_train_2dpca_reshaped = X_train_2dpca.reshape(X_train.shape[0], -1)
        # projection of test data on eigenvectors
        for i in range(n_test):
            X_test_2dpca[i] = np.dot(X_test_centered[i].T, eigenvectors)
        # reshape the 3D tensor of train and test data to 2D
        X_test_2dpca_reshaped = X_test_2dpca.reshape(X_test.shape[0], -1)
        # SVM classifier
        t_svm = time()
        param_grid = {
            "C": loguniform(1e3, 1e5),
            "gamma": loguniform(1e-4, 1e-1),
        }
        clf = RandomizedSearchCV(
            SVC(kernel="rbf", class_weight="balanced"), param_grid, n_iter=10
        )
        clf.fit(X_train_2dpca_reshaped, y2_train)
        print("-------------------------------------------------------")
        print(trail_name + ' 2D PCA with %d components' % n)
        print("time for SVM: %0.3fs" % (time() - t_svm))
        print("Best estimator found by grid search:")
        print(clf.best_estimator_)
        # evaluation
        target_names = ["Drone", "Bird"]
        y_pred = clf.predict(X_test_2dpca_reshaped)
        avg_acc.append(clf.score(X_test_2dpca_reshaped, y2_test))
        print("Classification report for classifier %s:\n%s\n"
              % (clf, classification_report(y2_test, y_pred, target_names=target_names)))
        ConfusionMatrixDisplay.from_estimator(
            clf, X_test_2dpca_reshaped, y2_test, display_labels=target_names, xticks_rotation="vertical"
        )
        # saved confusion matrix
        plt.tight_layout()
        plt.savefig('./outputs/2DPCA_with' + str(X_train_reshape.shape[1]) + 'bin' + 'CFM' + str(n) + '.png', dpi=300,
                    bbox_inches='tight')
        print("time of 2D PCA with %d components: %0.3fs" % (n, time() - t0))
        print("Accuracy: %0.3f" % clf.score(X_test_2dpca_reshaped, y1_test))
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
    plt.savefig('./outputs/2DPCA_with' + str(X_train_reshape.shape[1]) + 'bin' + '.png', dpi=300, bbox_inches='tight')
    plt.show()
    # Delete Variables and Release Memory
    del X, X_reshape, X_train, X_test, y1, y2, y1_train, y1_test, y2_train
    del y2_test, X_train_reshape, X_test_reshape, X_train_centered, X_test_centered
    del eigenvalues, eigenvectors, idx, X_train_2dpca, X_test_2dpca
    del X_train_2dpca_reshaped, X_test_2dpca_reshaped, cov_train, avg_train
    del param_grid, clf, target_names, y_pred, n_components
    gc.collect()
    # return avg_acc
    return avg_acc

### Conventional PCA on dataset with SVM classifier
import gc
import numpy as np
import matplotlib.pyplot as plt
from time import time
from scipy.stats import loguniform
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from dataset import dataset,createDataset


def conv_PCA_trail(trail_name,dir):
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

    # Standardization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # PCA with different number of components, from 5 to 50
    t0 = time()
    n_components = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    avg_acc = []
    for n in n_components:
        t0 = time()
        pca = PCA(n_components=n, svd_solver="randomized", whiten=True).fit(X_train)
        X_train_pca = pca.transform(X_train)
        X_test_pca = pca.transform(X_test)
        # SVM
        t_svc = time()
        param_grid = {
            "C": loguniform(1e3, 1e5),
            "gamma": loguniform(1e-4, 1e-1),
        }
        clf = RandomizedSearchCV(
            SVC(kernel="rbf", class_weight="balanced"), param_grid, n_iter=10
        )
        clf = clf.fit(X_train_pca, y2_train)
        print("-------------------------------------------------------")
        print(trail_name)
        print(' Conv PCA with %d components' % n)
        print('time for SVM: %0.3fs' % (time() - t_svc))
        print("Best estimator found by grid search:")
        print(clf.best_estimator_)
        # Evaluation, using pca data
        target_names = ["Drone", "Bird"]
        y_pred = clf.predict(X_test_pca)
        print(classification_report(y2_test, y_pred, target_names=target_names))
        ConfusionMatrixDisplay.from_estimator(
            clf, X_test_pca, y2_test, display_labels=target_names, xticks_rotation="vertical"
        )
        # save the confusion matrix
        plt.tight_layout()
        plt.savefig('./outputs/ConvPCA_with' + str(X.shape[2]) + 'bin' + 'CFM' + str(n) + '.png', dpi=300,
                    bbox_inches='tight')
        # Average accuracy
        avg_acc.append(clf.score(X_test_pca, y2_test))
        print("PCA with %d components done in %0.3fs" % (n, time() - t0))
        print("-------------------------------------------------------")

    # Print Figure of Accuracy vs Number of Components
    plt.figure()
    plt.plot(n_components, avg_acc, "b", label="Accuracy")
    plt.xlabel("Number of Components")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Number of Components")
    plt.legend()
    # Save the figure
    plt.savefig('./outputs/ConvPCA_with' + str(X.shape[2]) + 'bin' + '.png', dpi=300, bbox_inches='tight')
    plt.show()


    # Delete Variables and Release Memory
    del X, X_reshape, X_train, X_test, y1, y2, y1_train, y1_test, y2_train, y2_test, scaler, pca, X_train_pca, X_test_pca
    del param_grid, clf, target_names, y_pred, n_components
    del myDataset
    gc.collect()
    # return avg_acc
    return avg_acc
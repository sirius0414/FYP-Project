### No PCA on dataset with SVM classifier
import gc
import numpy as np
import matplotlib.pyplot as plt
from time import time
from scipy.stats import loguniform
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from dataset import createDataset

def without_PCA(name,dir):
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
    X_train, X_test, y1_train, y1_test, y2_train, y2_test = train_test_split(X_reshape, y1, y2, test_size=0.2, random_state=42,
                                                                             stratify=y1)
    # size of train and test data
    print("Size of train data: %d" % len(X_train))
    print("Size of test data: %d" % len(X_test))

    # Standardization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # SVM
    t0 = time()
    param_grid = {
        "C": loguniform(1e3, 1e5),
        "gamma": loguniform(1e-4, 1e-1),
    }
    clf = RandomizedSearchCV(
        SVC(kernel="rbf", class_weight="balanced"), param_grid, n_iter=10
    )
    clf = clf.fit(X_train, y2_train)
    print("-------------------------------------------------------")
    print(name)
    print("Best estimator found by grid search:")
    print(clf.best_estimator_)
    print('time for SVM: %0.3fs' % (time() - t0))

    # Evaluation, using pca data
    target_names = ["Drone", "Bird"]
    y_pred = clf.predict(X_test)
    print(classification_report(y2_test, y_pred, target_names=target_names))
    ConfusionMatrixDisplay.from_estimator(
        clf, X_test, y2_test, display_labels=target_names, xticks_rotation="vertical"
    )
    plt.tight_layout()
    # saved confusion matrix
    plt.savefig('./outputs/without_PCA' + str(X.shape[2]) + 'bin' + 'CFM' + '.png', dpi=300,
                bbox_inches='tight')
    plt.show()
    print('-------------------------------------------------------')
    # Delete variables and free memory
    del X, y1, y2, X_train, X_test, y1_train, y1_test, y2_train, y2_test
    del scaler, clf, param_grid, target_names, y_pred
    gc.collect()
    return 0


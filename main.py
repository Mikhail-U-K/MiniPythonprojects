from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn import datasets


def kneighbors():
    data_set = np.genfromtxt("wine.data", delimiter=',')
    y = data_set[:, 0]
    x = data_set[:, 1:]
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    lst = []
    k = 1
    while k < 51:
        neighbors_classifier = KNeighborsClassifier(n_neighbors=k)
        quality = cross_val_score(neighbors_classifier, x, y, cv=kf, scoring='accuracy')
        quality_mean = np.mean(quality)
        lst.append(quality_mean)
        print("k = ", k, " ", quality_mean)
        k += 1
    x_scales = scale(x)
    k = 1
    while k < 51:
        neighbors_classifier = KNeighborsClassifier(n_neighbors=k)
        quality = cross_val_score(neighbors_classifier, x_scales, y, cv=kf, scoring='accuracy')
        quality_mean = np.mean(quality)
        lst.append(quality_mean)
        print("new k = ", k, " ", quality_mean)
        k += 1


def k_neighbors_regressor():
    boston_data = datasets.load_boston()
    y = boston_data.target
    x = boston_data.data
    print(x)
    print(y)
    scale(x)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    param = np.linspace(1, 10, num=200)
    best_p = 0
    quality_best = 0
    i = 0
    while i < len(param):
        knr = KNeighborsRegressor(n_neighbors=5, weights='distance', p=param[i], metric='minkowski')
        quality_1 = cross_val_score(knr, x, y, cv=kf, scoring='neg_mean_squared_error')
        quality_mean_1 = np.min(quality_1)
        print("p: ", param[i], "i: ", i, quality_mean_1)
        if quality_best > quality_mean_1:
            quality_best = quality_mean_1
            best_p = param[i]
        i += 1
    print("p: ", best_p, " quality: ", quality_best)


def perceptron():
    test_data = pd.read_csv("data-test.csv", header=None)
    train_data = pd.read_csv("data-train.csv", header=None)
    y_test = test_data.iloc[:, 0]
    x_test = test_data.iloc[:, 1:]
    y_train = train_data.iloc[:, 0]
    x_train = train_data.iloc[:, 1:]
    clf = Perceptron()
    clf.fit(x_train, y_train)
    quality_before_scaling = accuracy_score(y_test, clf.predict(x_test))
    print("quality before", quality_before_scaling)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    clf.fit(x_train_scaled, y_train)
    quality_after_scaling = accuracy_score(y_test, clf.predict(x_test_scaled))
    print("quality after ", quality_after_scaling)
    print("result ", quality_after_scaling - quality_before_scaling)


def svc_for_data():
    data = pd.read_csv("svm-data.csv", header=None)
    y = data.iloc[:, 0]
    x = data.iloc[:, 1:]
    svc = SVC(kernel='linear', C=100000)
    svc.fit(x, y)
    y_pred = svc.predict(x)
    # print(y_pred)
    print(svc.support_ + 1)


if __name__ == 'main':
    print()

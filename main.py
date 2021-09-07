from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

if __name__ == '__main__':
    data_set = np.genfromtxt("wine.data", delimiter=',')
    y = data_set[:, 0]
    x = data_set[:, 1:]
    # print("classes ", y)
    # print("features ", x)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    k = 1
    while k < 51:
        neighbors_classifier = KNeighborsClassifier(n_neighbors=k)
        quality = cross_val_score(neighbors_classifier, x, y, cv=kf, scoring='accuracy')
        quality_mean = np.mean(quality)
        print("k = ", k, " ", quality_mean)
        k += 1

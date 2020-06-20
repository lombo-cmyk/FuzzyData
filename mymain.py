import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from skfuzzy.cluster import cmeans, cmeans_predict
from sklearn.cluster import KMeans
"""use crisp and fuzzy clustering
use at least 10 different division matrices at start (random choice)
Change parameters of fuzzy and crisp clustering -  In this way try to find the 
optimal solution with the lowest error for test data
 Make plots of cllusters projected on 2-dimansional surface. 
 Using Principal Component Analysis will be the most appreciated"""


def load_data():
    train_data = np.genfromtxt('ann-train.data', delimiter=' ')
    train_data = np.delete(train_data, slice(1, 16), 1)
    test_data = np.genfromtxt('ann-test.data', delimiter=' ')
    test_data = np.delete(test_data, slice(1, 16), 1)
    return train_data, test_data


def perform_PCA(training: np.array, test: np.array):
    pca = PCA(n_components=2)
    pca_ret = list()
    for dataset in [training, test]:
        pca.fit(dataset)
        pca_ret.append(pca.transform(dataset))
    return pca_ret[0], pca_ret[1]


def add_diagnosis_info(arr: np.array, diagnosis: np.array):
    ret = np.zeros((arr.shape[0], arr.shape[1]+1))
    ret[:, :-1] = arr
    ret[:, -1] = diagnosis
    return ret


def plot_results(data: np.array, title: type.__str__, no_clusters=3):
    for clust in range(no_clusters, 0, -1):
        plt.plot(data[data[:, 2] == clust, 0], data[data[:, 2] == clust, 1],
                 'o', markersize=3)
    plt.title(title)
    plt.show()


def perform_cmeans(train_set: np.array, test_set: np.array,
                   no_clusters=3, m=3):
    center, train_labels = cmeans(train_set.T, no_clusters, m, error=0.005,
                                  maxiter=1000, init=None)[0:2]
    test_labels = cmeans_predict(test_set.T, center, m, error=0.005,
                                 maxiter=1000)[0]
    test_clusters = np.argmax(test_labels, 0)
    test_clusters += 1
    train_clusters = np.argmax(train_labels, 0)
    train_clusters += 1
    return train_clusters.T, test_clusters.T


def perform_kmeans(train_set: np.array, test_set: np.array,
                   no_clusters=3):
    clustering_method = KMeans(no_clusters)
    clustering_method.fit(train_set)
    train_diagnosis = clustering_method.labels_ + 1
    test_diagnosis = clustering_method.predict(test_set) + 1
    return train_diagnosis, test_diagnosis


def main():
    train_data, test_data = load_data()

    train_2d, test_2d = perform_PCA(train_data[:, :-1], test_data[:, :-1])
    train_2d_with_diagnosis = add_diagnosis_info(train_2d, train_data[:, -1])
    test_2d_with_diagnosis = add_diagnosis_info(test_2d, test_data[:, -1])
    plot_results(train_2d_with_diagnosis, "training data - given")
    plot_results(test_2d_with_diagnosis, "test data - given")

    train_clusters, test_clusters = perform_cmeans(train_data[:, :-1],
                                                   test_data[:, :-1])
    train_data_cmeans_diagnosis = add_diagnosis_info(train_2d, train_clusters)
    test_data_cmeans_diagnosis = add_diagnosis_info(test_2d, test_clusters)
    plot_results(train_data_cmeans_diagnosis, "training results")
    plot_results(test_data_cmeans_diagnosis, "test results")

    train_clusters, test_clusters = perform_kmeans(train_data[:, :-1],
                                                   test_data[:, :-1])
    train_data_kmeans_diagnosis = add_diagnosis_info(train_2d, train_clusters)
    test_data_kmeans_diagnosis = add_diagnosis_info(test_2d, test_clusters)
    plot_results(train_data_kmeans_diagnosis, "training results kmeans")
    plot_results(test_data_kmeans_diagnosis, "test results kmeans")


if __name__ == "__main__":
    main()

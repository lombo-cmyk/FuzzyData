import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

"""use crisp and fuzzy clustering
use at least 10 different division matrices at start (random choice)
Change parameters of fuzzy and crisp clustering -  In this way try to find the 
optimal solution with the lowest error for test data
 Make plots of cllusters projected on 2-dimansional surface. 
 Using Principal Component Analysis will be the most appreciated"""

def load_data():
    train_data = np.genfromtxt('ann-train.data', delimiter=' ')
    test_data = np.genfromtxt('ann-test.data', delimiter=' ')
    return train_data, test_data

def perform_PCA(array: np.array):
    pca = PCA(n_components=2)
    pca.fit(array)
    return pca.transform(array)

def add_diagnosis_info(arr: np.array, diagnosis: np.array):
    arr = np.append(arr, diagnosis, axis=1)
    return arr

def plot_results(data: np.array, title: type.__str__, no_clusters=np.array([1,2,3])):
    for clust in no_clusters:
        plt.plot(data[data[:, 2] == clust, 0], data[data[:, 2] == clust, 1], 'o', markersize=3)
    plt.title(title)
    plt.show()
    plt.close()

def main():
    train_data, test_data = load_data()
    processed_arrays = list()
    processed_arrays.append(perform_PCA(train_data))
    processed_arrays.append(perform_PCA(test_data))
    processed_arrays[0] = add_diagnosis_info(processed_arrays[0], train_data[:, 21:22])
    processed_arrays[1] = add_diagnosis_info(processed_arrays[1], test_data[:, 21:22])
    plot_results(processed_arrays[0], "training data")
    plot_results(processed_arrays[1], "test data")



if __name__ == "__main__":
    main()
from sklearn.neighbors import KDTree
from kneed import KneeLocator
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter


class HaloBasedClustering:
    def __init__(
            self,
            n_neighbors=10,
            density_threshold_pct=50,
            tol=2,
            S=4,
            min_cluster=10,
            reallocation=True

    ):

        self.n_neighbors = n_neighbors
        self.density_threshold_pct = density_threshold_pct
        self.tol = tol
        self.S = S
        self.plot = False
        self.min_cluster = min_cluster
        self.reallocation = reallocation

    def haloDetection(self):
        DescendingOrder = sorted(self.rd)

        self.haloThreshold = np.percentile(np.unique(DescendingOrder), self.density_threshold_pct)
        self.halo = np.where(self.rd < self.haloThreshold)[0]
        self.core = np.where(self.rd >= self.haloThreshold)[0]

    def RelativeDensityEstimate(self, X, k):
        DBS, DDS, DGS = GNN_Decomposition(X, k)
        rd = (DGS + DBS * 0.5) / (DDS + DGS + DBS)
        rd = MinMaxScaler().fit_transform(rd.reshape(-1, 1))
        return rd.ravel()

    def fit(self, X):
        self.rd = self.RelativeDensityEstimate(X, self.n_neighbors)
        self.haloDetection()

    def fit_predeict(self, X):
        self.fit(X)
        self.haloIsolation(X)
        self.filter_small_clusters()

        if self.reallocation:
            self.noiseDetection()
            self.haloReallocation(X)

        return self.labels

    def filter_small_clusters(self):
        """
        Identifies and labels clusters with fewer than 10 data points as noise (-1).

        Args:
            labels (numpy.ndarray or list): An array of cluster labels.

        Returns:
            numpy.ndarray: A new array with small clusters relabeled as -1.  Returns None if input is invalid.

        """
        if not isinstance(self.labels, (np.ndarray, list)):
            print("Error: Input must be a NumPy array or a list.")
            return None

        label_counts = Counter(self.labels)  # Efficiently counts occurrences of each label
        small_clusters = [label for label, count in label_counts.items() if count < self.min_cluster]

        filtered_labels = np.copy(self.labels)  # Create a copy to avoid modifying the original array.
        self.cluster = self.core.copy()
        for label in small_clusters:
            self.cluster = np.setdiff1d(self.cluster, np.where(self.labels == label)[0])
            self.border = self.halo.copy()
            self.border = np.append(self.border, np.where(self.labels == label))
            filtered_labels[filtered_labels == label] = -1

        self.labels = filtered_labels

    def haloIsolation(self, X):
        m = X.shape[0]
        data_core = X[self.core]
        data_halo = X[self.halo]
        # TreeStucture
        halo_tree = KDTree(data_halo)
        core_tree = KDTree(data_core)

        self.labels = np.zeros(m, dtype=int) - 1
        core_rdorder_Index = self.core[np.argsort(self.rd[self.core])]

        # print('innerpoints_outIndex',innerpoints_outIndex)
        label_num = 1

        D, _ = halo_tree.query(data_core, k=self.tol, return_distance=True,
                               dualtree=False, breadth_first=False)
        D = D.max(axis=1)

        for i in core_rdorder_Index:
            if self.labels[i] != -1:
                continue

            stack = set([i])
            while stack:
                i = stack.pop()
                if self.labels[i] == -1:
                    self.labels[i] = label_num
                    x_i = X[i].reshape(1, -1)
                    di = D[self.core == i]
                    core_index_r = \
                        core_tree.query_radius(x_i, r=di, count_only=False, return_distance=True, sort_results=True)[0][
                            0]
                    neighbors = self.core[core_index_r[1:]]
                    neighbors = neighbors[self.labels[neighbors] == -1]
                    stack = stack.union(set(neighbors))

            label_num += 1

    def noiseDetection(self):
        self.rd_ascd = sorted(self.rd)
        xr = range(self.rd.shape[0])
        kneedle = KneeLocator(xr, self.rd_ascd, S=self.S, curve='concave', direction='increasing')
        self.knee_y = kneedle.knee_y
        self.knee = kneedle.knee
        self.y_difference = kneedle.y_difference

        try:
            # self.border = np.where((self.rd > kneedle.knee_y) & (self.rd < self.haloThreshold))[0]

            self.noise = np.where(self.rd <= kneedle.knee_y)[0]
            self.border = np.setdiff1d(self.border, self.noise)
        except TypeError as e:
            if 'NoneType' in str(e):
                # Handle the 'NoneType' exception
                return
            else:
                # Handle other types of exceptions
                raise e

    def _plot(self, X, cmap='viridis'):
        plt.figure(figsize=(5, 3.5), dpi=100)
        plt.plot(range(self.rd.shape[0]), self.rd_ascd, c='r', linewidth=1, label='Density curve')
        plt.plot(range(self.rd.shape[0]), self.y_difference, c='b', linewidth=1, label='Difference curve')

        # plt.text(knee.knee,knee.knee_y,s='(%d,%.2f)'%(knee.knee,knee.knee_y),fontsize=20,alpha=1,c='r')
        plt.plot([self.knee, self.knee], [0, 1], '--', color='k')
        plt.scatter(self.knee, self.knee_y, marker='x', s=100, c='k', linewidths=3, label='Knee point')
        plt.legend(fontsize=14)
        plt.ylabel('Relative density', fontsize=14)
        plt.xlabel('Points ordering', fontsize=14)

        plt.figure(figsize=(5, 3.5), dpi=100)
        plt.scatter(X[self.core][:, 0], X[self.core][:, 1], s=1, c='b', label='core')
        plt.scatter(X[self.border][:, 0], X[self.border][:, 1], s=(1 - self.rd[self.border]) ** 6 * 1000, marker='o',
                    c='w', alpha=0.9, edgecolors='k', linewidths=0.5, label='border')
        plt.scatter(X[self.noise][:, 0], X[self.noise][:, 1], s=30, marker='x', c='r', alpha=0.5, label='noise')
        plt.title('HaloModel', fontsize=16)
        plt.legend(fontsize=14)
        plt.axis('off')

        plt.figure(figsize=(5, 3.5), dpi=100)
        # plt.set_cmap('BrBG_r')
        current_cmap = plt.get_cmap(cmap)
        plt.scatter(X[self.noise, 0], X[self.noise, 1], s=25, marker='x', c='grey', linewidths=1, alpha=0.5,
                    label='noise')
        for i in pd.Series.value_counts(self.labels[self.cluster]).index:
            min = pd.Series.value_counts(self.labels[self.cluster]).index.min()
            max = pd.Series.value_counts(self.labels[self.cluster]).index.max()
            span = max - min
            plt.scatter(X[self.core[self.labels[self.core] == i], 0], X[self.core[self.labels[self.core] == i], 1],
                        c=current_cmap((i - min) / span), s=5, marker='o', edgecolors='k', linewidths=0.2, alpha=0.5,
                        label=f'Object from cluster {i}')
            plt.scatter(X[self.border[self.labels[self.border] == i], 0],
                        X[self.border[self.labels[self.border] == i], 1],
                        s=25, marker='o', c='w', edgecolors=current_cmap((i - min) / span), linewidths=1,
                        label=f'Halo for cluster {i}')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, fancybox=True, shadow=True)
        plt.title('3 layers')
        plt.axis('off')

    def haloReallocation(self, data):
        m = data.shape[0]

        # data_border = data[self.border]
        sorted_haloIndices = self.border[np.argsort(-self.rd[self.border])]

        for i in sorted_haloIndices:

            current_p = data[i]
            # coretree = KDTree(data[self.cluster])
            coretree = KDTree(data[self.cluster])
            k = 10
            D, ind = coretree.query(current_p.reshape(1, -1), k=k, return_distance=True, dualtree=False,
                                    breadth_first=False)
            if k == 1:
                self.labels[i] = self.labels[self.cluster[ind]]
            else:
                # self.labels[i] = pd.Series.value_counts(self.labels[self.cluster[ind]]).sort_values().index[-1]

                self.labels[i] = Counter(self.labels[self.cluster[ind][0]]).most_common()[0][0]
            # print(cluster.shape,i)
            self.cluster = np.append(self.cluster, i)


def GNN_Decomposition(data, k):
    kGraph = kneighbors_graph(data, k, n_jobs=-1).toarray().astype(int)
    rNNGraph = kGraph.T
    DBS = (kGraph & rNNGraph).sum(axis=1)
    DDS = (kGraph - (kGraph & rNNGraph)).sum(axis=1)
    DGS = (rNNGraph - (kGraph & rNNGraph)).sum(axis=1)
    return DBS, DDS, DGS
import numpy as np
from sklearn.cluster import KMeans

def run_kmeans(eigen_vecs_k, args):
    # ***********************************************************
    # ***********************************************************
    # CHANGE THIS TO A NATIVE PYTHON KMEANS IMPLEMENTATION
    # ***********************************************************
    # ***********************************************************
    kmeans = KMeans(n_clusters=args.num_clusters, random_state=0).fit(eigen_vecs_k)
    clustered_labels = kmeans.labels_

    unique, counts = np.unique(clustered_labels, return_counts=True)
    print(f'After K-Means\n{dict(zip(unique, counts))}')

    return clustered_labels
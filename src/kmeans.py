import numpy as np

def run_kmeans(eigen_vecs_k, args, n_iterations=50, eps=1e-7):
    '''
        k-means++ algorithm
        arguments:
         - eigen_vecs_k: np.ndarray of shape [no_data, no_dimensions]
                       input data points
         - args: arguments for number of clusters
         - n_iterations: integer, number of iterations to run k-means for
         - eps: another stopping criteria, if the centroids don't change much
        returns:
         - which_component: np.ndarray of shape [no_data] and integer data
                            type, contains values in [0, k-1] indicating which
                            cluster each data point belongs to
         - centroids:  np.ndarray of shape [k, no_dimensions], centres of
                       final custers, ordered in such way as indexed by
                       `which_component`
        '''
    k = args.num_clusters
    centroids = k_means_pp_initialization(eigen_vecs_k, k)
    for _ in range(n_iterations):
        # reassign data points to components
        distances = np.linalg.norm(np.expand_dims(eigen_vecs_k, axis=1) - centroids, axis=-1, ord=2)
        which_component = np.argmin(distances, axis=-1)
        # calcuate centroid for each component
        centroids_new = np.stack(list(eigen_vecs_k[which_component == i].mean(axis=0) for i in range(k)), axis=0)
        if (np.linalg.norm(centroids_new-centroids)<eps):
            break
        else:
            centroids = centroids_new
    return which_component, centroids


def k_means_pp_initialization(X, k):
    '''
    Compute initial cluster centers for k-means
    arguments:
     - X:          np.ndarray of shape [no_data, no_dimensions]
                   input data points
    returns:
     - centroids:  np.ndarray of shape [k, no_dimensions]
                   centres of initial clusters
    '''
    n, d = X.shape
    eps = 1e-8
    idx = np.array([np.random.randint(n)])
    centroids = X[idx, :].reshape((1, d))
    for i in range(k - 1):
        # find minimum distance to centroids
        min_d = np.min(np.linalg.norm(np.expand_dims(X, axis=1) - centroids, axis=-1, ord=2), axis=1)
        # handle degeneracy
        min_d[np.where(min_d) == 0] = eps
        # assign the older centroids idx as zero again -> we don't want to choose same point
        min_d[idx] = 0
        p = np.power(min_d, 2) / np.sum(np.power(min_d, 2))
        # choose and index with p probability
        chosen_idx = np.random.choice(n, 1, p=p)
        # update the idx -> add index of the chosen point, and centroids-> the chosen point
        idx = np.vstack((idx, chosen_idx))
        centroids = np.vstack((centroids, X[chosen_idx, :]))

    return centroids

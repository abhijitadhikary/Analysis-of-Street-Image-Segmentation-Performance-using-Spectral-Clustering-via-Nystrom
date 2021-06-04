from kmeans import run_kmeans

def get_clustered_image(eigen_vectors, args):
    '''
    Performs clustering on a supplied set of eigenvectors and returns the clustered image
    :param eigen_vectors: np.ndarray of shape [num_elements, num_clusters-1]
                        each column (eigen vector) has num_elements = width*height size
                        num_clusters - 1 -> the number of eigenvectors extracted for clustering
    :param args:  arguments for relevant hyper-parameters
    :return clustered_image: np.ndarray of shape [height, width]
                            labels from clustering reshaped to the shape of the image
            clustered_labels: np.ndarray of shape [num_elements,],  num_elements = width*height
                            labels from clustering
    '''
    # ***********************************************************
    # ***********************************************************
    # THINK OF USING SOMETHING OTHER THAN KMEANS, i.e. GMM
    # ***********************************************************
    # ***********************************************************

    # clustered_labels = run_kmeans_sklearn(image, args)
    clustered_labels, _ = run_kmeans(eigen_vectors, args)
    clustered_image = clustered_labels.reshape(args.height, args.width)

    return clustered_image

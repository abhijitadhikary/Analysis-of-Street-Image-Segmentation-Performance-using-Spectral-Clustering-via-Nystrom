from kmeans import run_kmeans_sklearn

def get_clustered_image(image, args):
    '''
    Performs clustering on a supplied image and returns the clustered image
    :param image:
    :param args:
    :return:
    '''
    # ***********************************************************
    # ***********************************************************
    # THINK OF USING SOMETHING OTHER THAN KMEANS, i.e. GMM
    # ***********************************************************
    # ***********************************************************

    clustered_labels = run_kmeans_sklearn(image, args)
    clustered_image = clustered_labels.reshape(args.height, args.width)

    return clustered_image, clustered_labels
import numpy as np
from time import time
from tqdm import tqdm
from scipy.linalg import sqrtm
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from sklearn.cluster import KMeans
import os
from utils import *

args = get_args()
image = cv2.imread('../notebooks/vegetables.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# image = get_dummy_image()
image = convert(image, min_value=0, max_value=1)

image_shape = len(image.shape)
if image_shape == 3:
    height, width, num_channels = image.shape
elif image_shape == 2:
    height, width = image.shape
    num_channels = 1

args.height = height
args.width = width
args.num_channels = num_channels
args.num_elements_flat = height * width
args.num_dimensions = num_channels + 2

image_array = get_image_array(image, args)
start = time()
eigen_vecs_k = get_k_eig_vectors_nystrom(image_array, args)
end = time()
print('Run time', end-start)
print(eigen_vecs_k.shape)
# imshow(eigen_vecs_k[:,0].reshape(args.height, args.width), '1st')
# imshow(eigen_vecs_k[:,1].reshape(args.height, args.width), '2nd')
# imshow(eigen_vecs_k[:,2].reshape(args.height, args.width), '3rd')
kmeans = KMeans(n_clusters=args.num_clusters, random_state=0).fit(eigen_vecs_k)
clustered_labels = kmeans.labels_
unique, counts = np.unique(clustered_labels, return_counts=True)
d = dict(zip(unique, counts))
print('After K-Means')
print(d)

clustered_image = clustered_labels.reshape(args.height, args.width)

# segment the image using average color for each cluster
segmented_image = get_segmented_image(image, clustered_image, clustered_labels, args)
unique, counts = np.unique(segmented_image, return_counts=True)
d = dict(zip(unique, counts))
print('After Segmentation')
print(d)
# imshow(image, 'Input Image')
# imshow(clustered_image, 'Clustered Image')
imshow(segmented_image, 'segmented_image')
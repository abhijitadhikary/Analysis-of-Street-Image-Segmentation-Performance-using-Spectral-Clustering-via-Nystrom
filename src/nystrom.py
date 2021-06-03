import numpy as np
from utils import *

def get_k_eig_vectors_nystrom(image_array, args):
    dim_low = args.dim_low

    indices_low = np.random.choice(args.num_elements_flat, size=dim_low, replace=False)
    image_low = image_array[indices_low]
    distances_colour = np.linalg.norm(np.expand_dims(image_array[:, :args.num_channels], axis=1) - image_low[:, :args.num_channels], axis=-1, ord=2)
    distances_position = np.linalg.norm(np.expand_dims(image_array[:, args.num_channels:], axis=1) - image_low[:, args.num_channels:], axis=-1, ord=2)
    weight_matrix = (get_exponential_bump(distances_colour, args.sigma_color) * get_exponential_bump(distances_position, args.sigma_distance)).T
    # weight_matrix = (get_exponential_bump(distances_colour, args.sigma_color)).T
    row = [i for i in range(dim_low)]
    weight_matrix[row, row] = 1

    A = weight_matrix[:, list(indices_low)]  # nxn
    B = np.delete(weight_matrix, list(indices_low), axis=1)  # nxm
    n, m = B.shape
    #     print(A.shape)
    #     print(B.shape)
    A_pinv = np.linalg.pinv(A)
    d1 = np.sum(np.vstack((A, B.T)), axis=0).reshape(1, -1)
    d2 = np.sum(B, axis=0) + (np.sum(B.T, axis=0).reshape(1, -1) @ (A_pinv @ B))
    dhat = np.sqrt(1 / np.hstack((d1, d2))).reshape(-1, 1)
    A = A * (dhat[:n].reshape(-1, 1) @ dhat[:n].reshape(-1, 1).T)
    B = B * (dhat[:n].reshape(-1, 1) @ dhat[n:].reshape(-1, 1).T)

    pinv_A = np.linalg.pinv(A)

    Asi = sqrtm(pinv_A)

    Q = A + Asi @ B @ B.T @ Asi
    U, L, T = np.linalg.svd(Q)

    L = np.diag(L)

    V = np.hstack((A, B)).T @ Asi @ U @ np.linalg.pinv(np.sqrt(L))

    V = V[:, 1:args.num_eigen_vectors]
    # reordering V appropriately
    all_idx = list(np.arange(args.num_elements_flat))
    rem_idx = [idx for idx in all_idx if idx not in indices_low]
    top_matrix = np.zeros((args.num_elements_flat, args.num_eigen_vectors-1))
    top_matrix[list(indices_low), :] = V[:dim_low, :]
    top_matrix[rem_idx, :] = V[dim_low:, :]
    return top_matrix
import numpy as np
from scipy.linalg import sqrtm

def run_nystrom(weight_matrix_partial, indices_random_low_dim):
    '''
    Calculates dim_low eigenvectors using the Nystrom on the weight marrix
    :param weight_matrix_partial:
    :param indices_random_low_dim:
    :return:
    '''
    # ***********************************************************
    # ***********************************************************
    # UPDATE THIS TO HAVE PROPER COMMENTS AND EXPLANATION
    # ***********************************************************
    # ***********************************************************
    A = weight_matrix_partial[:, list(indices_random_low_dim)]  # nxn
    B = np.delete(weight_matrix_partial, list(indices_random_low_dim), axis=1)  # nxm
    n, m = B.shape
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
    # ignore the comlpex portion if there is any
    V = np.real(V)
    return V
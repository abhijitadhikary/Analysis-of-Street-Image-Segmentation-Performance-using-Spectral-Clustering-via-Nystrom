import numpy as np
from scipy.linalg import sqrtm

def run_nystrom(weight_matrix_partial, indices_random_low_dim):
    '''
    Calculates dim_low eigenvectors using the Nystrom on the weight matrix.

    Our implementation follows the pseudocode provided in the following paper:
        Fowlkes, C., et al. “Spectral Grouping Using the Nystrom Method.
        ” IEEE Transactions on Pattern Analysis and Machine Intelligence,
        vol. 26, no. 2, Feb. 2004, pp. 214–25. IEEE Xplore,
        doi:10.1109/TPAMI.2004.1262185.
    A formal proof is also provided in the report methodology section, we reference the code to the
    equations in the report

    :param weight_matrix_partial: np.ndarray of shape [dim_low, num_elements]
                       dim_low -> the number of random pixels chosen for Nystorm
                       num_elements -> the total number of pixels in the image
    :param indices_random_low_dim: np.ndarray of shape [dim_low,]
                        the indices of the random pixels
    :return V: np.ndarray of shape [num_elements, dim_low]
                The eigenvectors for as approximated by nystorm method
                Each eigenvector is ordered w.r.t. the indices of the random pixels -> needs reordering
    '''
    # The sub-matrices inside the full weight matrix
    A = weight_matrix_partial[:, list(indices_random_low_dim)]  # nxn
    B = np.delete(weight_matrix_partial, list(indices_random_low_dim), axis=1)  # nxm
    n, m = B.shape
    A_pinv = np.linalg.pinv(A)
    # calculation approximate degree matrix from A and B (report Eq. 19)
    d1 = np.sum(np.vstack((A, B.T)), axis=0).reshape(1, -1)
    d2 = np.sum(B, axis=0) + (np.sum(B.T, axis=0).reshape(1, -1) @ (A_pinv @ B))
    dhat = np.sqrt(1 / np.hstack((d1, d2))).reshape(-1, 1)
    # normalizing A and B to consider the diagonal matrix (report Eq. 20)
    A = A * (dhat[:n].reshape(-1, 1) @ dhat[:n].reshape(-1, 1).T)
    B = B * (dhat[:n].reshape(-1, 1) @ dhat[n:].reshape(-1, 1).T)

    pinv_A = np.linalg.pinv(A)

    Asi = sqrtm(pinv_A)
    # This is the one shot method described in the 2004 paper, also S = G^TG as mentioned in the report Eq. 17
    Q = A + Asi @ B @ B.T @ Asi

    svd_converged = True
    try:
        U, L, T = np.linalg.svd(Q)
    except np.linalg.LinAlgError as error:
        svd_converged = False
    L = np.diag(L)
    # The orthogonal eigenvectors of approximated, as mentioned in the report Eq. 18
    V = np.hstack((A, B)).T @ Asi @ U @ np.linalg.pinv(np.sqrt(L))
    # ignore the complex portion if there is any
    V = np.real(V)
    return V, svd_converged
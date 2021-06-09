import numpy as np
from kmeans import run_kmeans as k_means
'''
    Most of the functions in this file is directly re-used from CLAB3
'''
def GMM_EM(x, init_mu, init_Sigma, init_pi, epsilon=0.001, maxiter=100):
    '''
    GMM-EM algorithm with shared covariance matrix
    arguments:
     - x:          np.ndarray of shape [no_data, no_dimensions]
                   input 2-d data points
     - init_mu:    np.ndarray of shape [no_components, no_dimensions]
                   means of Gaussians
     - init_Sigma: np.ndarray of shape [no_dimensions, no_dimensions]
                   covariance matrix of Gaussians
     - init_pi:    np.ndarray of shape [no_components]
                   prior probabilities of P(z)
     - epsilon:    floating-point scalar
                   stop iterations if log-likelihood increase is smaller than epsilon
     - maxiter:    integer scaler
                   max number of iterations
    returns:
     - mu:    np.ndarray of shape [no_components, no_dimensions]
              means of Gaussians
     - Sigma: np.ndarray of shape [no_dimensions, no_dimensions]
              covariance matrix of Gaussians
     - pi:    np.ndarray of shape [no_components]
              prior probabilities of P(z)
    '''
    mu = init_mu
    Sigma = init_Sigma
    pi = init_pi
    no_iterations = 0

    # compute log-likelihood of P(x)
    logp = np.log(incomplete_likelihood(x, mu, Sigma, pi)).sum()
    print("Init log P(x) is {:.4e}".format(logp))

    while True:
        no_iterations = no_iterations + 1

        # E step
        gamma = E_step(x, mu, Sigma, pi)
        # M step
        mu, Sigma, pi = M_step(x, gamma)

        # exit loop if log-likelihood increase is smaller than epsilon
        # or iteration number reaches maxiter
        new_logp = np.log(incomplete_likelihood(x, mu, Sigma, pi)).sum()
        if new_logp < epsilon + logp or no_iterations > maxiter:
            print("Iteration {:03} log P(x) is {:.4e}".format(no_iterations, new_logp))
            break
        else:
            print("Iteration {:03} log P(x) is {:.4e}".format(no_iterations, new_logp), end="\r")
            logp = new_logp

    return mu, Sigma, pi

def E_step(x, mu, Sigma, pi):
    '''
    arguments:
     - x:     np.ndarray of shape [no_data, no_dimensions]
              input 2-d data points
     - mu:    np.ndarray of shape [no_components, no_dimensions]
              means of Gaussians
     - Sigma: np.ndarray of shape [no_dimensions, no_dimensions]
              covariance matrix of Gaussians
     - pi:    np.ndarray of shape [no_components]
              prior probabilities of P(z)
    returns:
     - gamma: np.ndarray of shape [no_data, no_components]
              probabilities of P(z|x)
    '''
    p = complete_likelihood(x, mu, Sigma, pi)
    return p / p.sum(axis=-1 ,keepdims=True)

def M_step(x, gamma):
    '''

    arguments:
     - x:     np.ndarray of shape [no_data, no_dimensions]
              input 2-d data points
     - gamma: np.ndarray of shape [no_data, no_components]
              probabilities of P(z|x)
    returns:
     - mu:    np.ndarray of shape [no_components, no_dimensions]
              means of Gaussians
     - Sigma: np.ndarray of shape [no_dimensions, no_dimensions]
              covariance matrix of Gaussians
     - pi:    np.ndarray of shape [no_components]
              prior probabilities of P(z)
    '''
    N_k = gamma.sum(axis=0) # [no_components]
    mu = gamma.T @ x / N_k[: ,np.newaxis]
    pi = N_k / x.shape[0]

    _, no_dimensions = x.shape
    deviation = (np.expand_dims(x ,axis=1) - mu).reshape((-1, no_dimensions))  # [no_data*no_comp, no_dim]
    Sigma = deviation.T * gamma.flatten() @ deviation / gamma.sum()
    return mu, Sigma, pi

def incomplete_likelihood(x, mu, Sigma, pi):
    '''
    arguments:
     - x:     np.ndarray of shape [no_data, no_dimensions]
              input 2-d data points
     - mu:    np.ndarray of shape [no_components, no_dimensions]
              means of Gaussians
     - Sigma: np.ndarray of shape [no_dimensions, no_dimensions]
              covariance matrix of Gaussians
     - pi:    np.ndarray of shape [no_components]
              prior probabilities of P(z)
    returns:
     - px:    np.ndarray of shape [no_data]
              probabilities of P(x) at each data point in x
    '''
    p = complete_likelihood(x, mu, Sigma, pi)
    return p.sum(axis=-1)


def complete_likelihood(x, mu, Sigma, pi):
    '''
    arguments:
     - x:     np.ndarray of shape [no_data, no_dimensions]
              input 2-d data points
     - mu:    np.ndarray of shape [no_components, no_dimensions]
              means of Gaussians
     - Sigma: np.ndarray of shape [no_dimensions, no_dimensions]
              covariance matrix of Gaussians
     - pi:    np.ndarray of shape [no_components]
              prior probabilities of P(z)
    returns:
     - p:     np.ndarray of shape [no_data, no_components]
              joint probabilities of P(x,z)
    '''
    no_dimensions = x.shape[-1]
    # compute Gauss density
    deviation = (np.expand_dims(x ,axis=1) - mu)
    pdf = np.exp((-0.5 * deviation @ np.linalg.pinv(Sigma) * deviation).sum(axis=-1)) / \
          np.power( 2 *np.pi, 0.5 *no_dimensions) / np.sqrt(pdet(Sigma))
    return pdf * pi

def pdet(M):
    '''
    returns the pseudo-determinant of square matrix
    M should be positive semi-definite
    '''
    rank = np.linalg.matrix_rank(M)
    sigmas = np.linalg.svd(M, compute_uv=False) # by default numpy's svd has |U|=|V|=1
    return np.prod(sigmas[0:rank])


def run_gmm(x, args):
    ### k-means initialisation ######################
    which_component, init_mu = k_means(x, args, n_iterations=20)
    init_Sigma = np.cov(x - init_mu[which_component], rowvar=False)
    _, n_counts = np.unique(which_component, return_counts=True)
    init_pi = n_counts / x.shape[0]

    mu, Sigma, pi = GMM_EM(x, init_mu, init_Sigma, init_pi)
    clustered_image = incomplete_likelihood(x, mu, Sigma, pi)
    return clustered_image

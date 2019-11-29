import numpy as np
import matplotlib.pyplot as plt
from BIP_LWR.samplers.mhsampler import MHSampler
from BIP_LWR.moves.gaussian import GaussianMove


def log_post(x, y):
    """log posterior: 2D gaussian"""
    vect = np.array([x,y])
    # mean and covariance
    mu = np.array([4,1])
    cov = np.array([[3,1],[1,3]])
    return -0.5 * np.linalg.multi_dot([vect-mu, cov, vect-mu])


if __name__ == '__main__':
    """
    Sample from a 2D gaussian
    """
    ICs = {'x': 1, 'y':-1}

    # instantiate a sampler
    mcmc = MHSampler(log_post=log_post, ICs=ICs, cov=None, verbose=2, save_chain=False)
    # use a gaussian move
    mcmc.move = GaussianMove(param_info=mcmc.backend.param_info, cov=np.eye(2))
    mcmc.run(n_iter=4000, print_rate=1000)

    # diagnostics
    mcmc.trace_plots()
    mcmc.kde_plots()
    plt.show()

# -*- coding: utf-8 -*-

import numpy as np
from collections import OrderedDict
from BIP_LWR.moves.gaussian import GaussianMove

class DelCastFDMove(GaussianMove):
    """
    Move for Del Castillo FD using the inverse paramterisation for w

    Parameters
    ----------
    param_info: dict
        Dictionary of paramters and initial conditions. Format: {'x': 3, 'y': 10}
    cov: ndarray, int, or float
        Covariance matrix of RW proposal
    """

    def __init__(self, param_info, cov=None):
        super(DelCastFDMove, self).__init__(param_info, cov)

    def get_proposal(self, current_samples):
        """
        Proposal function for Del Castillo FD with inverse parametrisation for w

        Parameters
        ----------
        current_samples: dict
            Dictionary of current samples

        Returns
        -------
        new_samples: dict
            Dictionary of updated samples using a Gaussian move
            Truncated Gaussian update for w (ie: reject straight away if w<=0)
        log_hastings: float
            Value of log-hastings for the proposal
        """
        new_samples = OrderedDict()
        new_sample_list = np.dot(self.chol, np.random.normal(size=4)) + list(current_samples.values())
        # make sure 1/w is positive
        while new_sample_list[-1] <= 0:
            new_sample_list = np.dot(self.chol, np.random.normal(size=4)) + list(current_samples.values())
        for idx, param in enumerate(['z', 'rho_j', 'u', 'w']):
            new_samples[param] = new_sample_list[idx]

        return new_samples, 0

    def __str__(self):
        move_description = """Random walk for FD parameters using a Gaussian proposal
        Truncated Gaussian proposal for w (ie: reject straight away if w<=0)"""
        return move_description

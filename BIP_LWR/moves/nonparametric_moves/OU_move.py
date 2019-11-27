# -*- coding: utf-8 -*-

import numpy as np
from BIP_LWR.moves.mh import MHMove
from .utils import Hastings_term_nonparam

class OUMove(MHMove):
    """
    RW on function space using OU proposals

    Parameters
    ----------
    OU: Ornstein Uhlenbeck object (from ou_process.py module)

    prior_mean: ndarray or None (default None)
        If None, defaults to 0
    """

    def __init__(self, OU, prior_mean=None):
        self.omega = 0.1
        self.OU = OU
        if prior_mean is None:
            self.prior_mean = np.zeros(self.OU.N)
        else:
            self.prior_mean = prior_mean
        super(OUMove, self).__init__(self.get_proposal)

    def get_proposal(self, current_samples):
        if len(current_samples) != 1:
            raise ValueError("get_proposal() in OUMove only takes a single OU function at a time")
        fun_current = list(current_samples.values())[0]
        fun_new = self.OU.sample()*self.omega + np.sqrt(1-self.omega**2)*(fun_current - self.prior_mean) + self.prior_mean

        log_hastings = Hastings_term_nonparam(fun_current=fun_current, fun_new=fun_new,
                                    proposal_mean=self.prior_mean, precision=self.OU.Precision)

        return {list(current_samples.keys())[0]: fun_new}, log_hastings

    def __str__(self):
        return "OU move"

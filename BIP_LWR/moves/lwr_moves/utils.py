# -*- coding: utf-8 -*-
from collections import OrderedDict
import numpy as np
from copy import deepcopy
from BIP_LWR.tools.util import FD_neg_power



def rho_crit(FD, w_transf_type):
    """
    Returns the density that reaches capacity for Del_Castillo FD.
    Input: dictionary of FD parameters (this include the parameters rho_j, u, and w)
    """
    if w_transf_type == 'inv':
        w = 1/FD['w']
    elif w_transf_type == 'nat':
        w = FD['w']
    else:
        raise ValueError("w_transf_type should be either 'inv' or 'nat'")
    return FD['rho_j']/(1+FD['u']**(w/(w+1)))


def build_phi_del_Cast(FD_1, FD_2, t_crit, BC_type, dataset):
    """
    Build linear function to shift BCs. This is based on the proposed FD move

    Parameters
    ----------
    FD1, FD2: dict
        Dictionary of 2 fundamental diagrams. FD1: current FD, FD2: proposed FD
    t_crit: int
        time at which CF starts (value of t_crit is included)
    BC_type: str
        Either 'BC_outlet' or 'BC_inlet'
    dataset: str
        Either 'Sim' or 'DS1'
    """
    FD_1 = {k:v for k,v in FD_1.items()}
    FD_2 = {k:v for k,v in FD_2.items()}
    z_current, z_new = FD_1['z'], FD_2['z']
    rho_j_current, rho_j_new = FD_1['rho_j'], FD_2['rho_j']
    # w_current, w_new = FD_1['w'], FD_2['w']
    # parameters fitted to training data
    # For del Castillo on simulated data (Nov4). Parameters fit to training data in Nov4; run_6; S3
    if BC_type == 'BC_outlet':
        if dataset == 'Sim':
            beta1, beta2 = 0.00780496, 0.00222498 # Sim-40t: from Feb6; Run_4; S3 (beta=0.56)
            # beta1, beta2 = 0.00572584, 0.00188674 # from Feb3; Run_6; S3 (beta=0.2)
        elif dataset == 'DS1':
            beta1, beta2 = 0.00951237, 0.00288304 # from Feb4; Run_10, process 3; S3 (beta=1)
        else:
            raise ValueError("`dataset` either 'Sim' or 'DS1'")
        # beta1, beta2 = 0.00427224, 0.00130753
        # beta1, beta2, beta3 = 0.00395316, 0.00113736, 0.11578577
    elif BC_type == 'BC_inlet':
        if dataset == 'Sim':
            beta1, beta2 = 0.00616622, 0.00136533 # Sim-40t: from Feb6; Run_4; S3 (beta=0.56)
            # beta1, beta2 = 0.00315195, 0.00126064 # from Feb4; Run_6; S3 (beta=0.2)
        elif dataset == 'DS1':
            beta1, beta2 = 0.01109042, 0.00341119
        else:
            raise ValueError("`dataset` either 'Sim' or 'DS1'")
        # beta1, beta2 = 0.00586241, 0.00202265
        # beta1, beta2, beta3 = 0.00508745, 0.00160937, 0.28120868
    else:
        raise ValueError("BC_type must be either 'BC_outlet' or 'BC_inlet'")

    def phi(BC):
        shift_val = beta1*(z_new-z_current) + beta2*(rho_j_new-rho_j_current) #+ beta3*(w_new-w_current)
        BC_new = deepcopy(BC)
        # multiply time exp(shift_val) to get a linear shift in OU space
        BC_CF = BC_new[t_crit:]*np.exp(shift_val)
        BC_new = np.concatenate([BC_new[:t_crit], BC_CF])
        return BC_new
    return phi




def build_phi_Exp(FD_1, FD_2, t_crit, BC_type, dataset):
    """
    Build linear function to shift BCs. This is based on the proposed FD move
    For the Exponential FD

    Parameters
    ----------
    FD1, FD2: dict
        Dictionary of 2 fundamental diagrams. FD1: current FD, FD2: proposed FD
    t_crit: int
        time at which CF starts (value of t_crit is included)
    BC_type: str
        Either 'BC_outlet' or 'BC_inlet'
    dataset: str
        Either 'Sim' or 'DS1'
    """
    FD_1 = {k:v for k,v in FD_1.items()}
    FD_2 = {k:v for k,v in FD_2.items()}
    alpha_current, alpha_new = FD_1['alpha'], FD_2['alpha']
    if BC_type == 'BC_outlet':
        if dataset == 'Sim':
            coef = -0.24383851 # from Feb5; run_4; process 1 (beta=0.56)
            # coef = -0.19070776 # from Feb1; Run_5, process 1; S3 (beta=0.63)
        elif dataset == 'DS1':
            coef = -0.42876477 # Feb2; Run_15; process1; S3
        else:
            raise ValueError("`dataset` either 'Sim' or 'DS1'")
    elif BC_type == 'BC_inlet':
        if dataset == 'Sim':
            coef = -0.23614185 # from Feb5; run_4; process 1 (beta=0.56)
            # coef = -0.22624891 # from Feb1; Run_5, process 1; S3 (beta=0.63)
        elif dataset == 'DS1':
            coef = -0.29092067 # Feb2; Run_15; process1; S3
        else:
            raise ValueError("`dataset` either 'Sim' or 'DS1'")
    else:
        raise ValueError("BC_type must be either 'BC_outlet' or 'BC_inlet'")

    def phi(BC):
        shift_val = coef*(alpha_new-alpha_current)
        BC_new = deepcopy(BC)
        # multiply time exp(shift_val) to get a linear shift in OU space
        BC_CF = BC_new[t_crit:]*np.exp(shift_val)
        BC_new = np.concatenate([BC_new[:t_crit], BC_CF])
        return BC_new
    return phi

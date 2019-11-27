# -*- coding: utf-8 -*-

import numpy as np
from copy import deepcopy
from functools import wraps
import time
import os
import copy
import boto3
import botocore
import pickle

from statsmodels.tsa.stattools import acf

from BIP_LWR.distributions.priors import prior_bounds
from BIP_LWR import config


def int_ACT(x, M=50):
    """
    Integrated autocorrelation time: T_int = 1 + sum \rho_j
    with \rho_j normalised autocorrelation function (uses statsmodels' function)

    Parameters
    ----------
    x: ndarray 
        MCMC samples
    M: int
        Cuttoff to estimate ACF. According to Sokal, M should be smallest value such that  M >= C*T_int.
        Choose C between 5 and 10.
    """
    # remove the first value of the ACF as this is 1
    acf_arr = acf(x, nlags=M)[1:]
    return 1 + 2*np.sum(acf_arr)

def BC_noise(BC, BC_mean, omega, chol_precision):
    """
    Preconditioned RW for BC. Use this to try random initial conditions for MCMC
    Input and output as BC are exp(OU)

    Parameters
    ----------
    BC: ndarray
        BC to apply noise to
    BC_mean: ndarray
        Prior mean for BC (in log(OU) parametrisation)
    omega: float
        Omega value for preconditioned RW
    chol_precision: ndarray
        Cholesky of prior precision matrix

    Returns
    -------
    new_BC: ndarray
        New BC with noise
    """
    N = len(chol_precision)
    sam_prior = np.linalg.solve(chol_precision, np.random.normal(loc=0, scale=1, size=N))
    log_current_BC = np.log(BC)
    log_BC_mean = np.log(BC_mean)
    log_new_BC = (sam_prior)*omega + np.sqrt(1-omega**2)*(log_current_BC - log_BC_mean) + log_BC_mean
    new_BC = np.exp(log_new_BC)
    return new_BC

def apply_FD_noise(FD, FD_par, FD_sigma):
    "Apply Gaussian noise to some FD parameters"
    new_param = np.random.normal(FD[FD_par], scale=FD_sigma)
    while not (prior_bounds[FD_par][0] < new_param < prior_bounds[FD_par][1]):
        new_param = np.random.normal(FD[FD_par], scale=FD_sigma)
    FD[FD_par] = new_param
    return FD

def sample_from_FD_prior():
    "Sample from the FD prior"
    FD = {k: np.random.uniform(low=v[0], high=v[1]) for k,v in prior_bounds.items()}
    return FD

def del_Cast_random_IC(FD_mean, mcmc_obj, omega, FD_noise_dict, FD_prior=False):
    """
    Returns initial conditions for a Del Castillo sampler with noise in the FD and the BCs
    """
    mcmc = deepcopy(mcmc_obj)
    FD = deepcopy(FD_mean)
    FD['BC_outlet'] = BC_noise(BC=FD['BC_outlet'], BC_mean=mcmc.move.BC_prior_mean['BC_outlet'],
                omega=omega, chol_precision=mcmc.move.BC_move.OU.cholPrecision)
    FD['BC_inlet'] = BC_noise(BC=FD['BC_inlet'], BC_mean=mcmc.move.BC_prior_mean['BC_inlet'],
                omega=omega, chol_precision=mcmc.move.BC_move.OU.cholPrecision)

    if FD_prior == False:
        # random variation of FD within prior bounds
        for FD_par, FD_sigma in FD_noise_dict.items():
            FD = apply_FD_noise(FD=FD, FD_par=FD_par, FD_sigma=FD_sigma)
    elif FD_prior == True:
        for k,v in sample_from_FD_prior().items():
            FD[k] = v
    # repeat if rho_j < BCs
    if FD['rho_j'] < max(max(FD['BC_outlet']), max(FD['BC_inlet'])):
        print("\n\nRho_j bigger than BCs\n\n")
        FD['BC_outlet'] = BC_noise(BC=FD['BC_outlet'], BC_mean=mcmc.move.BC_prior_mean['BC_outlet'],
                    omega=omega, chol_precision=mcmc.move.BC_move.OU.cholPrecision)
        FD['BC_inlet'] = BC_noise(BC=FD['BC_inlet'], BC_mean=mcmc.move.BC_prior_mean['BC_inlet'],
                    omega=omega, chol_precision=mcmc.move.BC_move.OU.cholPrecision)
        if FD_prior == False:
            for FD_par, FD_sigma in FD_noise_dict.items():
                FD = apply_FD_noise(FD=FD, FD_par=FD_par, FD_sigma=FD_sigma)
        elif FD_prior == True:
            for k,v in sample_from_FD_prior().items():
                FD[k] = v

    del mcmc
    return FD

def scale_cov_directions(cov, alpha_list=[1,1,1,1]):
    """
    Scale a covariance matrix in all 4 directions

    Parameters
    ----------
    cov: ndarray
        Covariance matrix
    alpha_list: list
        List of factors to scale each eigenvalue
    """
    evals, evects = np.linalg.eig(cov)
    scaled_evals = deepcopy(evals)
    for idx in range(4):
        scaled_evals[idx] = scaled_evals[idx]*alpha_list[idx]
    new_cov = np.linalg.multi_dot([evects, np.diag(scaled_evals), np.linalg.inv(evects)])
    return new_cov

def scale_omega_gibbs(alpha, omega, omega_max=0.51):
    """
    Scale omega parameter in section_dict Gibbs blocks.
    Use a linear fuction:
    new_omega = (omega_0 - omega_max)*(alpha**0.2) + omega_max
    """
    if not (0<=alpha<=1):
        raise ValueError("alpha parameter in SA should be between 0 and 1")
    return round((omega - omega_max)*(alpha**0.2) + omega_max,10)

def scale_gibbs_blocks(alpha, section_dict, omega_max=0.51):
    """
    Scale BC Gibbs blocks based on inverse temperature parameter alpha

    Parameters
    section_dict: dict
        Dictionary of section dict parameters.
        Format:
        section_dict = {
            'section_1': {'param': 'BC_inlet', 'cut1': 0, 'cut2':150, 'omega': 0.1},
            'section_2': {'param': 'BC_outlet', 'cut1':0, 'cut2':150, 'omega':0.1},
            }

    Returns
    -------
    section_dict: dict
        Section_dict with modified 'w' values for each section
    """
    new_section_dict = deepcopy(section_dict)
    for k,v in new_section_dict.items():
        new_section_dict[k]['omega'] = scale_omega_gibbs(alpha=alpha, omega=v['omega'], omega_max=omega_max)
    return new_section_dict

def load_CSV_data(path):
    """
    Load CSV from the data folder (`traffic_data`)

    Parameters
    ----------
    path: str
        Path relative to `traffic_data` folder

    Returns
    -------
    data: ndarray
        Array of data
    """
    return np.genfromtxt(os.path.join(config.DATA_PATH, path))

def build_schedule_array(len_schedule, num_plateaux, initial_temp):
    """
    Builds and array of Simulated Annealing schedule parameters
    Uses f(t)=t**6 as schedule function (with t in [0,1])
    The last element of the step_schedule array is always 1

    Parameters
    ----------
    len_schedule: int
        Number of iterations of schedule
    num_plateaux: int
        Number of num_plateaux of constant temperature in schedule
    initial_temp: float
        Starting point for temperature schedule

    Returns
    -------
    step_schedule: ndarray
        Array of schedule parameters
    """
    t_array = np.linspace(0, 1, len_schedule)
    idx_step = int(len_schedule/num_plateaux)

    le_fun = lambda x: (x**6)*(1-initial_temp) + initial_temp
    schedule_array = [le_fun(elem) for elem in t_array]
    step_schedule = np.concatenate([[schedule_array[elem*idx_step]]*idx_step for elem in range(num_plateaux)])

    # make sure that the last element of the array is always 1
    if len(step_schedule)==len_schedule:
        step_schedule[-1]=1
    elif len(step_schedule)<len_schedule:
        step_schedule = np.append(step_schedule, [1]*(len_schedule - len(step_schedule)))
    else:
        raise ValueError("Prolem with lenghts somewhere")
    return step_schedule


def transf_w(w, w_transf_type='inv'):
    if w_transf_type == 'log_inv':
        return np.log(1/w)
    elif w_transf_type == 'nat':
        return w
    elif w_transf_type == 'inv':
        return 1/w
    else:
        raise ValueError("w_transf_type must be 'nat', 'inv', or 'log_inv'")


def time_it(fun):
    """
    Decorator that returns execution time of function
    if time > 60s: returns the time in minutes as well as seconds
    """
    @wraps(fun)
    def _wrapper(*args, **kwargs):
        start = time.time()
        result = fun(*args, **kwargs)
        end = time.time()
        time_in_sec = end-start
        time_in_min = np.floor((time_in_sec)/60).astype('int')
        num_sec = (time_in_sec) % 60
        if time_in_sec > 60:
            min_str = "({0} min {1} sec)".format(time_in_min, int(num_sec))
            time_in_sec = int(time_in_sec)
        else:
            min_str = ''
            time_in_sec = round(time_in_sec,3)
        print("Running time: {0} sec {1}".format(time_in_sec, min_str))
        return result
    return _wrapper



def FD_neg_power(rho, w, u, rho_j, z):
    """
    Del Castillo FD.

    Parameters:
    ----------
    rho: density
    w, u, rho_j, z: FD parameters
    """
    return z * ( (u*rho/rho_j)**(-w) + (1-rho/rho_j)**(-w) )**(-1/w)

def FD_exp(rho,alpha,beta):
    """
    V(rho) = a*exp(-b*rho)
    """
    return alpha*np.exp(-beta*rho)*rho

def load_pickled_file(file):
    """
    Loads data from a pickled file.
    """

    f = open(file, 'rb')
    data = pickle.load(f)
    f.close()
    return data

def upload_chain(s3_path, local_path, bucket_name='lwr-inverse-us-east'):
    """
    Upload file to S3

    Parameters
    ----------
    s3_path: str
        Path to file in S3
    local_path: str
        Path to file on local machine
    bucket_name: str 
        Either 'lwr-inverse-mcmc' or 'lwr-inverse-us-east' (the latter is default)
    """
    s3 = boto3.resource("s3")
    lwr_inv = s3.Bucket(bucket_name)
    file_content = open(local_path, 'rb')
    lwr_inv.put_object(Key=s3_path, Body=file_content)

def download_chain(s3_path, local_path, bucket_name='lwr-inverse-us-east'):
    """
    Download file from S3. Uses s3 bucket 'lwr-inverse-mcmc'

    Parameters
    ----------
    s3_path: str
        Path to file in S3
    local_path: str
        Path to file on local machine
    bucket_name: str 
        Either 'lwr-inverse-mcmc' or 'lwr-inverse-us-east'  (the latter is default)
    """
    s3 = boto3.resource("s3")
    lwr_inv = s3.Bucket(bucket_name)
    try:
        lwr_inv.download_file(Key=s3_path, Filename=local_path)
        print("Download successful")
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
                print("The object does not exist.")
        else:
            raise

def create_latex_table_section_dict(section_dict, mcmc_vis):
    """
    Function to create the inner portion of a latex table of Gibbs blocks fo the appendix.
    Prints the output

    Parameters
    ----------

    section_dict: dict 
        Dictionary of gibbs blocks
    mcmc_vis: mcmc_vis object


    """
    for k,v in section_dict.items():
        accept_list = []
        for chain_num in [1,2,3]:
            accept_list.append(str(round(mcmc_vis.accept_Gibbs(chain_num=chain_num, section_num=int(k[8:])),1)))
        accept_r = ", ".join(accept_list)
        line1 = v['param'][3:] + " & " + str(v['cut1'])  + " & " + str(v['cut2']) + " & " + str(v['omega']) + " & "  
        print(line1 + accept_r + " \\\ [1ex]\n \hline")
        

def create_latex_matrix(cov):
    """
    Create latex code for {pmatrix}
    """
    print("Matrix:\n{}".format(cov))
    print("\n=====\nLatex:\n")
    for elem in cov:
        row_str = " & ".join([str(e) for e in elem])
        print(row_str + " \\\ ")
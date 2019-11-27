# -*- coding: utf-8 -*-

import numpy as np
import os
import datetime, calendar
from copy import deepcopy

from BIP_LWR.samplers.mhsampler import MHSampler
from BIP_LWR.backends.lwr_backend import LWRBackend
from BIP_LWR.lwr.lwr_solver import LWR_Solver
from BIP_LWR.config import LWRConfigHolder
from BIP_LWR.tools.BC_ICs import BC_outlet_IC_1, BC_inlet_IC_1

from BIP_LWR.tools.util import scale_gibbs_blocks
from BIP_LWR.moves.lwr_moves.lwr_FD_BCGibbs import LWR_FD_BCGibbs

class LWRSampler(MHSampler):
    """
    Sampler for LWR

    Parameters
    ----------
    ICs: dict
        format: {'x': IC}
    config_dict: dict
        Dictionary of configuration parameters
    cov: ndarray, int/float, or None
        Covariance matrix of proposal. If None: uses an identity matrix.
    verbose: int
        0: print nothing
        1: print basic info when running.
        2: print basic info when running along with acceptance rate
    data_array : ndarray or None
        array of raw data (3 columns) to use to compute the loss
        If None: use data_array in config.py
    """
    def __init__(self, ICs, config_dict, cov=None, verbose=1, data_array=None):
        # remove BCs in ICs so the gaussian proposal in mhsampler doesn't complain
        FD_params = {k:v for k,v in ICs.items() if k not in ['BC_outlet', 'BC_inlet']}
        super(LWRSampler, self).__init__(log_post=self.log_posterior, ICs=FD_params,
                                        cov=cov, verbose=verbose)
        self.backend = LWRBackend(ICs)
        self.lwr = LWR_Solver(data_array=data_array, config_dict=config_dict)
        self.config = LWRConfigHolder(**config_dict)
        self.BC_outlet_IC = np.genfromtxt(os.path.join(self.config.DATA_PATH, "BCs", "BC_outlet_sample_2.csv"))
        self.BC_inlet_IC = np.genfromtxt(os.path.join(self.config.DATA_PATH, "BCs", "BC_inlet_sample_2.csv"))
        self.solver = "le_solver"


    def log_prior_BC(self, BC, mean):
        """
        # Note: mean is already an OU mean (in lwr_FD_BCGibbs.py)
        BC and mean should be exp(OU) processes
        """
        BC = np.log(BC)
        mean = np.log(mean)
        return -0.5*np.linalg.multi_dot([BC-mean, self.move.OU.Precision, BC-mean])

    def log_prior(self, *args, **kwargs):
        return 0

    def log_posterior(self, BC_outlet=BC_outlet_IC_1, BC_inlet=BC_inlet_IC_1, *args, **kwargs):
        return - self.lwr.loss_function(solver=self.solver, BC_outlet=BC_outlet, BC_inlet=BC_inlet, *args, **kwargs) + self.log_prior(*args, **kwargs) \
        + self.log_prior_BC(BC=BC_inlet, mean=self.move.BC_prior_mean['BC_inlet']) + self.log_prior_BC(BC=BC_outlet, mean=self.move.BC_prior_mean['BC_outlet'])

    def alpha_schedule(self):
        """
        Schedule for Simulated Annealing.
        Override this in subclasse

        Returns
        -------
        alpha_temp: float
            Parameter to multiplty log-likelihood by
        alpha_changed: Bool
            Whether or not alpha_temp has just changed
        """
        return 1, False

    def step(self):
        """
        Run a step of MCMC

        1. Propose samples and accept/reject
        2. Update backend on which Gibbs section were sampled
        3. Save parameters to backend
        """
        # propose and accept/reject
        new_samples, loss_new, accepted = self.move.propose(current_samples=self.backend.current_samples,
            loss_current=self.backend.loss_current, log_posterior=self.log_posterior)

        # update backend on current gibbs section (unless `self.move` doesn't have the `current_gibbs` attribute)
        try:
            if accepted == True:
                self.backend.update_sample_params("{}_a".format(self.move.current_gibbs_param))
                # if the accepted sample was beta_temp, update the covariance matrices
                if self.move.current_gibbs_param == 'beta_temp':
                    self.move.update_cov_tempering(current_samples=self.backend.current_samples)
            else:
                self.backend.update_sample_params("{}_r".format(self.move.current_gibbs_param))
            self.backend.update_current_section(self.move.BC_move.current_section)
        except AttributeError:
            pass

        # save step to backend
        self.backend.save_step(new_samples=new_samples, loss_new=loss_new, accepted=accepted)


    def progress_bar(self, iter_num, n_iter, print_rate, init=False):
        current_samples = {k:v for k, v in self.backend.current_samples.items() if k not in ['BC_inlet', 'BC_outlet']}

        if init==True:
            if type(self.config.DATA_TRANSFORM)==type(lambda x:x):
                data_transform = 'no_transform'
            string_print_1 = "\nRunning MCMC for {}\n======================".format(self.__str__())
            string_print_2 = "\nRun {}".format(self.config.RUN_NUM)
            string_print_3 = "\nInitial conditions: {0} = {1}"\
                .format(list(current_samples.keys()), [round(x,2) for x in list(current_samples.values())])
            string_print_4 = "\nData: {}".format(self.config.data_array_dict['flow'])
            string_print_5 = "\nComments: {}\n".format(self.config.comments)

            # print(string_print_1+string_print_2+string_print_3+string_print_4+string_print_5+"\n")

            if self.verbose>=1:
                print("Running MCMC for {} iterations...".format(n_iter))
                print("\n"+"-"*10)
            else:
                pass
        if self.verbose>=1:
            if iter_num >= print_rate:
                if iter_num%print_rate==0:
                    print("Iteration {0}/{1}. Samples: {2}: {3}). Log-post: {4}"\
                    .format(iter_num, n_iter, list(current_samples.keys()), [round(x,2) for x in list(current_samples.values())] , round(self.backend.log_post_list[-1],2)))
        else:
            pass

    def save_to_file(self, iter_num, iter_step):
        """
        Save to file every 'iter_step' number of samples

        Parameters
        ----------
        iter_num: int
            Current iteration number
        iter_step: int
            Save chain at every multiple of iter_step
        move_probs: list 
            List of move probability. This is to drop the BCs if they aren't being sampled (ie: if move_probs==[1,0,0])
        """
        if iter_num%iter_step==0:
            current_date = datetime.datetime.now()
            year = current_date.year
            month = calendar.month_name[current_date.month]
            day = current_date.day

            attrs = {'cov': self.move.cov,
                    'move': self.move.__str__(),
                    'date': "{} {} {}".format(day, month, year),
                    'folder_and_run': self.config.my_analysis_dir + "-Run_" + str(self.config.RUN_NUM),
                    'data_array_dict': str(self.config.data_array_dict),
                    'upload_to_S3': str(self.config.upload_to_S3),
                    'comments': self.config.comments,
                    'step_save': self.config.step_save,
                            }
            if hasattr(self.move, 'cov_joint'):
                attrs.update({'cov_joint': self.move.cov_joint})
            else:
                pass
            self.backend.to_file(solver=self.solver, config=self.config, 
                            move_probs=self.move.move_probs, attrs=attrs)
        else:
            pass


    def fit_cov(self, params):
        """
        Fit covariance matrix to all samples for some chosen parameters in the FD
        """
        df_samples = self.all_samples[params]
        return np.cov(df_samples.values.T)

    def __str__(self):
        return "Base LWR sampler"

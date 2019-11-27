 # -*- coding: utf-8 -*-

import os, shutil
import h5py
import numpy as np
import pandas as pd
from collections import OrderedDict

from .backend import Backend
from BIP_LWR.tools.util import upload_chain

import warnings
warnings.filterwarnings('ignore', category=pd.io.pytables.PerformanceWarning)

class LWRBackend(Backend):
    """
    Backend for LWR sampler

    Parameters
    ----------
    ICs: dict
        format: {'x': IC}
    """


    def __init__(self, ICs):
        super(LWRBackend, self).__init__(ICs)
        # 'list_current_section' keeps track of which sections were sampled (in an OU process for example)
        # 'list_sample_params' keeps track of accept/rejects (ex: "FD_a", "FD_r", "BC_a", "BC_r")
        self.gibbs_info = {'list_current_section': ['IC'],
                            'list_sample_params': ['IC']
                            }



    def update_sample_params(self, current_gibbs_param):
        self.gibbs_info['list_sample_params'].append(current_gibbs_param)

    def update_current_section(self, current_section):
        self.gibbs_info['list_current_section'].append(current_section)

    def to_file(self, solver, config, move_probs, attrs=None):
        """
        Save dataframe of samples & log_post to hdf5 in the appropriate directory

        Parameters
        ----------
        solver: str
            Name of solver (ie: which FD)
        config: LWRConfigHolder object
            Object holding configuration info (ie: where to save the MCMC chain)
        attrs: None or dict
            Dictionary of attributes to save in hdf5 file
        """
        param_info = OrderedDict([(k, v) for k, v in self.param_info.items() if k not in ['BC_inlet', 'BC_outlet']])
        FD_initial_conditions = OrderedDict([(k, v) for k, v in self.initial_conditions.items() if k not in ['BC_inlet', 'BC_outlet']])

        # Build directory name
        BCs = OrderedDict([(k, v) for k, v in self.current_samples.items() if k in ['BC_inlet', 'BC_outlet']])
        if BCs:
            BC_print_str = 'sampling_BCs_'+'-'.join([str(k[3:]) for k in list(BCs.keys())])
        else:
            BC_print_str = ""

        if config.upload_to_S3 == True:
            root_path = os.path.join(os.path.abspath('../Analysis'), '2018/tmp_S3_directory')
        else:
            root_path = os.path.abspath(config.MCMC_OUTPUT)

        # main_path = config.DATA_VARIABLE+"_{0}""-70108_{1}_{2}".format(config.ERROR_MODEL, solver, BC_print_str)
        # main_path = os.path.join(main_path, "run_"+str(config.RUN_NUM))
        # only include the run rumber, not the data_variable, error model, or data day
        main_path = "run_"+str(config.RUN_NUM)
        if hasattr(self, 'proc_num'):
            # add another folder if running a PTSampler
            main_path = os.path.join(main_path, 'process_{}'.format(self.proc_num))
        else:
            pass
        if config.upload_to_S3 == True:
            main_path = os.path.join(config.my_analysis_dir, main_path)
        else:
            pass
        hdf5_path = os.path.join(root_path, main_path)
        if not os.path.exists(hdf5_path):
            os.makedirs(hdf5_path)

        # Build file name
        str_IC = '-'.join(["_".join([k, str(v)]) for k,v in zip(list(param_info.keys()), FD_initial_conditions.values())])
        MCMC_csv = 'Samples-IC_{0}.csv'.format(str_IC)
        # create dataframe of samples. Save only every `save_step` sample so the hdf5 file isn't too big
        df = self.chain_to_df().iloc[::config.step_save]
        # remove BCs if they are not being sampled
        if (move_probs == [1, 0, 0]) or (config.FD_only==True): 
            df.drop(['BC_outlet', 'BC_inlet'], axis=1, inplace=True)
        # RSGS for FD and BC parameters
        if len(self.gibbs_info['list_sample_params'])>1:
            df['param_accept'] = self.gibbs_info['list_sample_params'][::config.step_save]
        # When sampling BCs in blocks
        if len(self.gibbs_info['list_current_section'])>1:
            df['BC_Gibbs'] = self.gibbs_info['list_current_section'][::config.step_save]

        # save to hdf5
        hdf5_full_path = os.path.join(hdf5_path, MCMC_csv[:-4]+".h5")
        df.to_hdf(path_or_buf=hdf5_full_path, key='MCMC_run', mode='w')
        # add attributes to file
        if attrs is not None:
            with h5py.File(hdf5_full_path, 'a') as f:
                f.create_group("attributes")
                # ==============
                # Count accepts and rejects for Parallel tempering
                array_gibbs, counts_gibbs = np.unique(self.gibbs_info['list_sample_params'], return_counts=True)
                if 'PT_beta_a' in array_gibbs:
                    acc_idx = list(array_gibbs).index('PT_beta_a')
                    num_accepts = counts_gibbs[acc_idx]
                else:
                    num_accepts = 0

                if 'PT_beta_r' in array_gibbs:
                    rej_idx = list(array_gibbs).index('PT_beta_r')
                    num_rejects = counts_gibbs[rej_idx]
                else:
                    num_rejects = 0
                f['attributes'].create_dataset('PT_accept_rejects', data=np.array([num_accepts, num_rejects]))
                # ==============
                f['attributes'].create_dataset('cov', data=attrs['cov'])
                if 'cov_joint' in attrs.keys():
                    f['attributes'].create_dataset('cov_joint', data=attrs['cov_joint'])
                # add all key-value pairs in attrs that aren't 'cov' and 'cov_joint'
                # as 'cov' and 'cov_joint' are numerical data, and everything else are strings
                attrs_key_list = [k for k in attrs.keys() if k not in ['cov', 'cov_joint']]
                for key in attrs_key_list:
                    f.attrs[key] = attrs[key]

        if config.upload_to_S3 == True:
            s3_path = os.path.join(main_path, MCMC_csv[:-4]+".h5")
            upload_chain(s3_path=s3_path, local_path=hdf5_full_path)
            # remove temporary local directory with hdf5 files
            # shutil.rmtree(root_path)

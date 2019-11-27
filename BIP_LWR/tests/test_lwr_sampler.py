# -*- coding: utf-8 -*-

import numpy as np
from collections import OrderedDict

from BIP_LWR.samplers.lwr.delcast_sampler import DelCastSampler
from BIP_LWR.tools.BC_ICs import BC_outlet_IC_2, BC_inlet_IC_2, BC_inlet_sample_prior_short, BC_outlet_sample_prior_short
from BIP_LWR.tools.MCMC_params import section_dict_70108_49t, cov_inv, cov_inv_joint
from BIP_LWR.tools.util import transf_w
from BIP_LWR.config import default_config_dict

import unittest


class TestLWRSampler(unittest.TestCase):

    def setUp(self, ratio_times_BCs=1):
        self.cov = cov_inv
        self.cov_joint = cov_inv_joint
        self.BC_outlet = BC_outlet_sample_prior_short
        self.BC_inlet = BC_outlet_sample_prior_short
        self.section_dict = section_dict_70108_49t
        self.move_probs = [1, 0, 0]
        self.config_dict =  {'my_analysis_dir': '2018/Default_directory',
                            'run_num': 1,
                            'data_array_dict': {'flow': 'test_data/artificial_data_array_flow_prior_BC_poisson_short.csv',
                                            'density': 'test_data/artificial_data_array_density_prior_BC_short.csv'},
                            'w_transf_type': 'inv',
                            'save_chain':False,
                            'ratio_times_BCs': ratio_times_BCs,
                              }

    def create_DelCast_sampler(self):
        z, rho_j, u, w = [170, 450, 3, 10]
        FD = OrderedDict([('z', z), ('rho_j', rho_j), ('u', u), ('w', transf_w(w=w, w_transf_type=self.config_dict['w_transf_type'])),
                ('BC_outlet', self.BC_outlet), ('BC_inlet', self.BC_inlet)])
        mcmc = DelCastSampler(ICs=FD, cov=self.cov, cov_joint=self.cov_joint, section_dict=self.section_dict,
                            move_probs=self.move_probs, config_dict=self.config_dict)
        return mcmc


    def test_del_Cast_sampler_wrong_order(self):
        """
        Create a MCMC sampler for del Castillo FD with the wrong ordering of parameters
        """
        z, rho_j, u, w = [170, 450, 3, 10]
        FD = OrderedDict([('rho_j', rho_j),('z', z),  ('u', u), ('w', transf_w(w=w, w_transf_type=self.config_dict['w_transf_type'])),
                ('BC_inlet', self.BC_inlet), ('BC_outlet', self.BC_outlet)])
        self.assertRaises(ValueError, DelCastSampler,
                            ICs=FD, cov=self.cov, cov_joint=self.cov_joint, section_dict=self.section_dict,
                            move_probs=self.move_probs, config_dict=self.config_dict)

    def test_BC_log_prior(self):
        """
        Create a MCMC sampler for del Castillo FD and calculate the log probability of
        some boundary condition
        """
        mcmc = self.create_DelCast_sampler()
        outlet_log_prior = round(mcmc.log_prior_BC(BC=BC_outlet_sample_prior_short, mean=mcmc.move.BC_prior_mean['BC_outlet']), 3)
        inlet_log_prior = round(mcmc.log_prior_BC(BC=BC_inlet_sample_prior_short, mean=mcmc.move.BC_prior_mean['BC_inlet']), 3)
        self.assertEqual(outlet_log_prior, -27.596)
        self.assertEqual(inlet_log_prior, -28.612)


    def test_run_delCast_sampler(self):
        """
        Simply run the Del Castillo sampler
        """
        mcmc = self.create_DelCast_sampler()
        mcmc.run(1,1)
        self.assertEqual(mcmc.config.save_chain, False)
        self.assertEqual(list(mcmc.all_samples.columns), ['z', 'rho_j', 'u', 'w', 'BC_outlet', 'BC_inlet', 'log_post'])

    def test_delCast_sampler_BCs_wrong_shape(self):
        """
        Create a Del Castillo sampler with BC shape not matching data shape.
        Running it should raise a ValueError
        """
        z, rho_j, u, w = [170, 450, 3, 10]
        FD = OrderedDict([('z', z), ('rho_j', rho_j), ('u', u), ('w', transf_w(w=w, w_transf_type=self.config_dict['w_transf_type'])),
                ('BC_outlet', BC_outlet_IC_2), ('BC_inlet', BC_inlet_IC_2)])
        mcmc = DelCastSampler(ICs=FD, cov=self.cov, cov_joint=self.cov_joint, section_dict=self.section_dict,
                            move_probs=self.move_probs, config_dict=self.config_dict)
        self.assertRaises(ValueError, mcmc.run, n_iter=1, print_rate=1)

    def test_BC_len_OU_size(self):
        """
        lwr.BC_len should be the same size as the BCs and the OU precision matrix
        """
        mcmc = self.create_DelCast_sampler()
        assert mcmc.move.OU.N == mcmc.lwr.BC_len
        assert mcmc.move.OU.N == mcmc.lwr.final_time * mcmc.lwr.config.ratio_times_BCs + 1


    def test_highdimensional_stuff(self):
        """
        Given some high dimensional BCs, check a bunch of lengths, such as:
        - N attribute in OU is correct
        - Size of Precision matrix in OU is correct
        - lwr.BC_len is correct
        - LWR lengths and data_to_PDE_times.values()
        """
        rt = 7
        self.setUp(ratio_times_BCs=rt)
        mcmc = self.create_DelCast_sampler()
        assert mcmc.lwr.final_time == 19
        assert mcmc.lwr.BC_len == mcmc.lwr.final_time * rt + 1
        assert mcmc.lwr.BC_len == mcmc.move.OU.N
        assert mcmc.lwr.BC_len == mcmc.move.OU.Precision.shape[0]
        assert mcmc.lwr.BC_len == mcmc.move.BC_prior_mean['BC_outlet'].shape[0]
        assert mcmc.lwr.BC_len == mcmc.move.BC_prior_mean['BC_inlet'].shape[0]

        # Check lwr stuff
        list_PDE_time = list(mcmc.lwr.data_to_PDE_time.values())
        list_PDE_time.sort()
        PDE_to_time_array = np.array(list_PDE_time)
        assert np.array_equal(np.arange(0, mcmc.lwr.final_time+1, 1), PDE_to_time_array)
        assert np.array_equal(mcmc.lwr.out_times, np.linspace(0, mcmc.lwr.final_time*rt, mcmc.lwr.final_time+1, endpoint=True))

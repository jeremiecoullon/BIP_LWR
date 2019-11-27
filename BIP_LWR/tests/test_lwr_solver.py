# -*- coding: utf-8 -*-

import os
import unittest
import numpy as np
from BIP_LWR.lwr.lwr_solver import LWR_Solver
from BIP_LWR import config
from BIP_LWR.tools.BC_ICs import BC_inlet_sample_prior_short, BC_outlet_sample_prior_short


class TestLWRSolver(unittest.TestCase):
    """
    Tests for the LWR_Solver class
    """

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def density_setup(self, ratio_times_BCs=1):
        """
        Setup to use in Tests_LWR_Solver
        """
        config_dict = {'my_analysis_dir': '2018/Default_directory',
                        'run_num': 1,
                        'data_array_dict': {'flow': 'test_data/artificial_data_array_flow_prior_BC_poisson_2.csv',
                                        'density': 'test_data/artificial_data_array_density_prior_BC.csv'},
                        'ratio_times_BCs': ratio_times_BCs
                      }
        self.LWR = LWR_Solver(config_dict=config_dict)

    def call_LWR_test(assert_error, FD):
        """
        Test Poisson error on flow data
        """
        def LWR_test_wrapper(self):
            """
            Wrapper for LWR tests with a given data variable and error model.
            """
            config_dict = {'my_analysis_dir': '2018/Default_directory',
                            'run_num': 1,
                            'data_array_dict': {'flow': 'test_data/artificial_data_array_flow_prior_BC_poisson_short.csv',
                                                'density': 'test_data/artificial_data_array_density_prior_BC_short.csv'
                                                },
                                        #     {'flow': 'data_array_flow_70108_longer.csv',
                                        # 'density': 'data_array_70108_density_longer.csv'}
                            'ratio_times_BCs': 1
                          }
            LWR = LWR_Solver(data_array=None, config_dict=config_dict)
            FD['solver'] = 'lwr_del_Cast'
            FD['BC_inlet'] = BC_inlet_sample_prior_short
            FD['BC_outlet'] = BC_outlet_sample_prior_short
            loss_LWR = LWR.loss_function(**FD)
            assert round(loss_LWR, 0) == assert_error
        return LWR_test_wrapper

    # run LWR loss_function tests
    # difference with linux: 4975.583/4975.531, 5823.793/5823.73, 6124.11/6124.325
    # test_LWR_del_Cast_flow_poisson_1 = call_LWR_test(4976, FD={'z': 170, 'rho_j': 444, 'u': 3.5105,'w': 10})
    test_LWR_del_Cast_flow_poisson_1 = call_LWR_test(859.0, FD={'z': 170, 'rho_j': 444, 'u': 3.5105,'w': 10})

    # test_LWR_del_Cast_flow_poisson_2 = call_LWR_test(5824, FD={'z': 160, 'rho_j': 440, 'u': 3,'w': 5})
    test_LWR_del_Cast_flow_poisson_2 = call_LWR_test(719.0, FD={'z': 160, 'rho_j': 440, 'u': 3,'w': 5})

    # test_LWR_del_Cast_flow_poisson_3 = call_LWR_test(6124, FD={'z': 200, 'rho_j': 400, 'u': 3,'w': 5})
    test_LWR_del_Cast_flow_poisson_3 = call_LWR_test(1114.0, FD={'z': 200, 'rho_j': 400, 'u': 3,'w': 5})


    def test_process_BCs_pass_None(self):
        """
        If you pass None to process_BCs(), should return self.BC_outlet/inlet.
        """
        self.density_setup()
        BC_outlet, BC_inlet = self.LWR.process_BCs(BC_outlet=None, BC_inlet=None)
        assert np.array_equal(BC_outlet, self.LWR.BC_outlet)
        assert np.array_equal(BC_inlet, self.LWR.BC_inlet)

    def test_process_BC_pass_array_correct_length(self):
        """
        If you pass arrays of the correct length to process_BCs(), they should
        return unchanged
        """
        self.density_setup()
        # pass 'native' LWR BCs
        BC_outlet, BC_inlet = self.LWR.process_BCs(BC_outlet=self.LWR.BC_outlet, BC_inlet=self.LWR.BC_inlet)
        assert np.array_equal(BC_outlet, self.LWR.BC_outlet)
        assert np.array_equal(BC_inlet, self.LWR.BC_inlet)

        # pass arbitrary BCs of correct length
        artificial_outlet = np.linspace(20, 84, self.LWR.final_time+1)
        artificial_inlet = np.linspace(20, 84, self.LWR.final_time+1)
        BC_outlet, BC_inlet = self.LWR.process_BCs(BC_outlet=artificial_outlet, BC_inlet=artificial_inlet)
        assert np.array_equal(BC_outlet, artificial_outlet)
        assert np.array_equal(BC_inlet, artificial_inlet)

    def test_LWR_correct_number_of_spaces(self):
        """
        Test data should have 8 detectors
        """
        self.density_setup()
        assert np.array_equal(self.LWR.data_spaces, np.array([0. , 1. , 2. , 2.5, 3. , 4. , 4.5, 5. ]))

    def test_LWR_correct_BCs(self):
        """
        Test that both BCs are correct for the test data
        """
        self.density_setup()
        LWR_test_inlet_BC = np.genfromtxt(os.path.join(config.DATA_PATH, "test_data/LWR_test_inlet_BC.csv"))
        LWR_test_outlet_BC = np.genfromtxt(os.path.join(config.DATA_PATH, "test_data/LWR_test_outlet_BC.csv"))
        assert np.array_equal(self.LWR.BC_inlet, LWR_test_inlet_BC)
        assert np.array_equal(self.LWR.BC_outlet, LWR_test_outlet_BC)

    def test_BC_len_and_final_time(self):
        """
        Check basic time/length stuff:
        - self.BC_len should be equal to length of BCs
        - final time should be the (centered) last time in lwr.data_times
        """
        self.density_setup()
        assert self.LWR.BC_len == self.LWR.BC_inlet.shape[0]
        assert self.LWR.BC_len == 150
        assert self.LWR.final_time == self.LWR.data_times[-1] - self.LWR.data_times[0]
        assert self.LWR.final_time == 149
        assert np.array_equal(self.LWR.out_times, np.linspace(0, 149, 150, endpoint=True))

    def test_ratio_times_BCs(self):
        """
        Check that length of BCs and data_to_PDE_time are correct 
        """
        rt = 16
        self.density_setup(ratio_times_BCs=rt)
        assert self.LWR.BC_len == self.LWR.final_time*rt + 1
        assert self.LWR.final_time == 149
        PDE_to_time_array = np.array(list(self.LWR.data_to_PDE_time.values()))
        assert np.array_equal(np.arange(0, self.LWR.final_time+1, 1), PDE_to_time_array)
        assert np.array_equal(self.LWR.out_times, np.linspace(0, self.LWR.final_time*rt, self.LWR.final_time+1, endpoint=True))




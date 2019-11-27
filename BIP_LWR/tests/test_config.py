# -*- coding: utf-8 -*-

import unittest
import os
from BIP_LWR.config import DelCastConfigHolder

class TestConfigHolder(unittest.TestCase):

    def test_config_variables(self):
        config_dict = {'my_analysis_dir': '2018/Default_directory',
                            'run_num': 3,
                            'data_array_dict': {'flow': 'test_data/artificial_data_array_flow_prior_BC_poisson_2.csv',
                                            'density': 'test_data/artificial_data_array_density_prior_BC.csv'},
                            'upload_to_S3': True,
                            'save_chain': False,
                            'w_transf_type': 'nat'
                              }
        config = DelCastConfigHolder(**config_dict)
        self.assertEqual(config.save_chain, False)
        self.assertEqual(config.w_transf_type, 'nat')
        self.assertEqual(config.upload_to_S3, True)
        self.assertEqual(config.RUN_NUM, 3)
        self.assertEqual(os.path.basename(config.BASE_PATH), 'BIP_LWR')

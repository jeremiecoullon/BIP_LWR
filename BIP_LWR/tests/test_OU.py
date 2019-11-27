# -*- coding: utf-8 -*-

import numpy as np
from BIP_LWR.moves.nonparametric_moves.OU_move import OUMove
from BIP_LWR.gp.ou_process import OUProcess
from BIP_LWR.moves.nonparametric_moves.utils import build_precision_blocks
import unittest


class TestOU(unittest.TestCase):

    def setUp(self):
        OU = OUProcess(**{'beta': 7, 'dt': 0.05,'sigma': 1.1, 'N':100})
        self.move = OUMove(OU)

    def test_OUMove_get_proposal_wrong_N(self):
        current_sample = {'OU_1': np.arange(10)}
        self.assertRaises(ValueError, self.move.get_proposal, current_sample)

    def test_OUMove_get_proposal_two_OU_objects(self):
        current_sample = {'OU_1': np.arange(100), 'OU_2': 10*np.arange(100)}
        self.assertRaises(ValueError, self.move.get_proposal, current_sample)

    def test_OUMove_get_proposal_returns_correct_shape(self):
        current_sample = {'OU_1': np.arange(100)}
        new_sample, hastings = self.move.get_proposal(current_sample)
        self.assertEqual(new_sample.keys(), current_sample.keys())


class TestOUGibbs(unittest.TestCase):

    def setUp(self):
        OU_params={'beta': 1, 'dt': 0.01, 'sigma': 10}
        self.N = 400
        IC = np.zeros(self.N)
        self.OU = OUProcess(N=self.N, **OU_params)

    def test_build_precision_blocks(self):
        """
        Test that the blocks in the precision matrix are correct
        """
        N_y1=10
        N_y2=129
        N_y3=self.N-N_y1-N_y2
        blocks = build_precision_blocks(Precision=self.OU.Precision, N_y1=N_y1, N_y2=N_y2, N_y3=N_y3)
        P_top = np.concatenate([blocks['P_11'], blocks['P_12'], blocks['P_13']], axis=1)
        P_middle = np.concatenate([blocks['P_21'], blocks['P_22'], blocks['P_23']], axis=1)
        P_bottom = np.concatenate([blocks['P_31'], blocks['P_32'], blocks['P_33']], axis=1)
        P_new = np.concatenate([P_top, P_middle, P_bottom], axis=0)
        assert np.array_equal(P_new, self.OU.Precision)

# -*- coding: utf-8 -*-

import numpy as np
from collections import OrderedDict
from .lwr_FD_BCGibbs import LWR_FD_BCGibbs
from .del_Cast_FD_move import DelCastFDMove
from .utils import build_phi_del_Cast
# from .helper_moves.independent_sampler_move import propose_new_BC_independent_move
# from .helper_moves.FD_GMM_move import GMM_get_proposal, shift_BCs

class DelCastMove(LWR_FD_BCGibbs):

    def __init__(self, param_info, section_dict, cov, cov_joint, config_dict,
            move_probs=[0.1, 0.2, 0.7], w_transf_type='inv'):
        super(DelCastMove, self).__init__(FDMove=DelCastFDMove, param_info=param_info, section_dict=section_dict,
                cov=cov, cov_joint=cov_joint, config_dict=config_dict, move_probs=move_probs)


    def propose_FD(self, new_samples, current_FD, current_BCs, FD_move, joint_BC_move=False):
        """
        Conditional FD|BC move and joint FD;BC move
        """
        # propose FD parameters
        new_FD, log_hastings = FD_move.get_proposal(current_FD)
        for k, v in new_FD.items():
            new_samples[k] = v
        # =========
        # fix FD to test the independent sampler
        # new_FD = current_FD
        # log_hastings = 0
        # =========
        if joint_BC_move == True:
            do_shift = False
            # keep BCs fixed
            if do_shift == False:
                # raise ValueError("Not shifting BCs!")
                new_samples['BC_outlet'] = current_BCs['BC_outlet']
                new_samples['BC_inlet'] = current_BCs['BC_inlet']
            else:
                raise ValueError("Shifting BCs!")
                pass
            # ==================
            # hacky joint shift (used with FD GMM)
            # trying out a GMM move for the FD
            # del new_FD, log_hastings
            # new_FD, log_hastings = GMM_get_proposal(current_samples=current_FD)
            # for k, v in new_FD.items():
            #     new_samples[k] = v
            # new_samples['BC_outlet'] = shift_BCs(BC_current=current_BCs['BC_outlet'], BC_type="BC_outlet",
            #                                     current_FD=current_FD, new_FD=new_FD)
            # new_samples['BC_inlet'] = shift_BCs(BC_current=current_BCs['BC_inlet'], BC_type="BC_inlet",
            #                                     current_FD=current_FD, new_FD=new_FD)
            # ==================
            # Good "shift" joint move
            # 70108, longer time: t_crit = 37,42 (for outlet, inlet)

            if do_shift == True:
                raise ValueError("Shifting BCs!")
                if self.lwr.config.data_array_dict['flow'] == 'data_array_70108_flow_49t.csv':
                    t_crit_outlet = 0
                    t_crit_inlet = 0
                    dataset = 'DS1'
                elif self.lwr.config.data_array_dict['flow']=='test_data/Simulated_LWR_Nov2018/data_array_DelCast_flow_RT_40.csv':
                    t_crit_outlet = 0
                    t_crit_inlet = 0
                    dataset = 'Sim'
                elif self.lwr.config.data_array_dict['flow']=='data_array_70108_flow_shorter.csv':
                    # for test_moves.py
                    t_crit_outlet = 0
                    t_crit_inlet = 0
                    dataset = 'Sim'
                elif self.lwr.config.data_array_dict['flow']=='Sim_data_array_DelCast_flow_40t.csv':
                    # Simulated data, 40t
                    t_crit_outlet = 0
                    t_crit_inlet = 0
                    dataset = 'Sim'
                else:
                    raise ValueError("t_crit in joint FD;BC move isn't defined for this dataset")
                t_crit_outlet = t_crit_outlet*self.lwr.config.ratio_times_BCs
                t_crit_inlet = t_crit_inlet*self.lwr.config.ratio_times_BCs
                phi_outlet = build_phi_del_Cast(FD_1=current_FD, FD_2=new_FD, t_crit=t_crit_outlet, BC_type='BC_outlet', dataset=dataset)
                phi_inlet = build_phi_del_Cast(FD_1=current_FD, FD_2=new_FD, t_crit=t_crit_inlet, BC_type='BC_inlet', dataset=dataset)
                new_samples['BC_outlet'] = phi_outlet(current_BCs['BC_outlet'])
                new_samples['BC_inlet'] = phi_inlet(current_BCs['BC_inlet'])
            else:
                pass
            # ==================

            # propose from Gaussian
            # new_samples['BC_outlet'], log_h_outlet = propose_BC_given_FD(BC_type='BC_outlet',
            #                 FD_current=current_FD, FD_new=new_FD, BC_current=current_BCs['BC_outlet'])
            # new_samples['BC_inlet'], log_h_inlet = propose_BC_given_FD(BC_type='BC_inlet',
            #                 FD_current=current_FD, FD_new=new_FD, BC_current=current_BCs['BC_inlet'])
            # log_hastings += log_h_inlet + log_h_outlet
            # identity:
            # new_samples['BC_outlet'] = current_BCs['BC_outlet']
            # print("Indepent sampler move")
            # new_samples['BC_outlet'], log_hastings = propose_new_BC_independent_move(outlet_BC_current=current_BCs['BC_outlet'])
            # new_samples['BC_inlet'] = current_BCs['BC_inlet']

            # ==================
            # independent sampler move for all parameters
            # new_samples = OrderedDict({})
            # all_current_params = np.concatenate([list(current_FD.values()), current_BCs['BC_outlet'], current_BCs['BC_inlet']])
            # all_params, log_hastings = propose_new_BC_independent_move(outlet_BC_current=all_current_params)
            # for idx, v in enumerate(['z', 'rho_j', 'u','w']):
            #     new_samples[v] = all_params[idx]
            # new_samples['BC_outlet'] = all_params[4:64]
            # new_samples['BC_inlet'] = all_params[64:]
        elif joint_BC_move == False:
            pass
        return new_samples, log_hastings

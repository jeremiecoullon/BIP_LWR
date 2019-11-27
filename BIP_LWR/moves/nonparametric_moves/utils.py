# -*- coding: utf-8 -*-

import numpy as np
from copy import deepcopy

def Hastings_term_nonparam(fun_current, fun_new, proposal_mean, precision):
    """
    Hastings correction for OU RW
    """
    new_inner = np.dot(fun_new.T, np.dot(precision, fun_new))
    current_inner = np.dot(fun_current.T, np.dot(precision, fun_current))
    current_mean_inner = np.dot(fun_current.T, np.dot(precision, proposal_mean))
    new_mean_inner = np.dot(fun_new.T, np.dot(precision, proposal_mean))
    return -0.5*(current_inner - new_inner - 2*current_mean_inner + 2*new_mean_inner)

# helper functions
def build_precision_blocks(Precision, N_y1, N_y2, N_y3):
    blocks = {
        'P_11': Precision[0:N_y1, 0:N_y1],
        'P_12': Precision[0:N_y1, N_y1:N_y1+N_y2],
        'P_13': Precision[0:N_y1, N_y1+N_y2::],
        'P_21': Precision[N_y1:N_y1+N_y2, 0:N_y1],
        'P_22': Precision[N_y1:N_y1+N_y2, N_y1:N_y1+N_y2],
        'P_23': Precision[N_y1:N_y1+N_y2, N_y1+N_y2::],
        'P_31': Precision[N_y1+N_y2::, 0:N_y1],
        'P_32': Precision[N_y1+N_y2::, N_y1:N_y1+N_y2],
        'P_33': Precision[N_y1+N_y2::, N_y1+N_y2::],
    }
    return blocks


# conditional mean: Y2|Y1,Y3
def build_cond_mean(y1, y3, mean1, mean2, mean3, block_22_21, block_22_23):
    term_1 = np.linalg.multi_dot([block_22_21, y1-mean1])
    term_2 = np.linalg.multi_dot([block_22_23, y3-mean3])
    return mean2 - term_1 - term_2



def create_N_y(section_dict, N):
    """
    Parameters
    ----------
    section_dict: dictionary
        Must have the keys: cut1 cut2

    N: int
        Total number of cells
    """
    for k,v in section_dict.items():
        section_dict[k]['N_y1']= v['cut1']
        section_dict[k]['N_y2']= v['cut2'] - v['cut1']
        section_dict[k]['N_y3']=N-v['cut2']
    return section_dict

def build_section_info(N_y1, N_y2, N_y3, prior_mean, OU_precision):
    blocks = build_precision_blocks(Precision=OU_precision, N_y1=N_y1, N_y2=N_y2, N_y3=N_y3)
    proposal_Precision = blocks['P_22']

    block_22_21 = np.dot(np.linalg.inv(blocks['P_22']), blocks['P_21'])
    block_22_23 = np.dot(np.linalg.inv(blocks['P_22']), blocks['P_23'])

    # 3 blocks for the prior mean
    mean1 = prior_mean[0:N_y1]
    mean2 = prior_mean[N_y1:(N_y1+N_y2)]
    mean3 = prior_mean[(N_y1+N_y2)::]

    chol_cond = np.linalg.cholesky(blocks['P_22']).T
    sect_dict = {}
    sect_dict['blocks']=blocks
    sect_dict['proposal_Precision']=proposal_Precision
    sect_dict['block_22_21']=block_22_21
    sect_dict['block_22_23']=block_22_23
    sect_dict['mean1']=mean1
    sect_dict['mean2']=mean2
    sect_dict['mean3']=mean3
    sect_dict['chol_cond']=chol_cond

    return sect_dict

def transform_section_dict(section_dict, ratio_times):
    """
    Transforms 'cut1' and 'cut2' from minutes to BC index based on ratio_times

    Parameters
    ----------
    section_dict: dict
        Section dict for Gibbs blocks.
        Format: `section_dict = {'section_1': {'param': 'x', 'cut1':0, 'cut2':100, 'omega':0.3},
                                'section_2': {'param': 'x', 'cut1':80, 'cut2':200, 'omega':0.3},
                                'section_3': {'param': 'y', 'cut1':0, 'cut2':200, 'omega':0.2},
                                }`
    ratio_times: int
        Ratio: N/final_time, with N number of time points in BC and final_time the number of minutes in the BCs
    """
    section_dict = deepcopy(section_dict)
    for k in section_dict.keys():
        section_dict[k]['cut1'] = section_dict[k]['cut1'] * ratio_times
        section_dict[k]['cut2'] = section_dict[k]['cut2'] * ratio_times - (ratio_times - 1)
    return section_dict

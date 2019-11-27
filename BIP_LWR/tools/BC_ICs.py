# -*- coding: utf-8 -*-

import numpy as np
import os
from BIP_LWR import config
from BIP_LWR.tools.util import load_CSV_data

# BC_outlet_IC_1 = np.genfromtxt(os.path.join(config.DATA_PATH, "BCs", "BC_outlet_sample_2.csv"))
# BC_inlet_IC_1 = np.genfromtxt(os.path.join(config.DATA_PATH, "BCs", "BC_inlet_sample_2.csv"))
BC_outlet_IC_1 = load_CSV_data("BCs/BC_outlet_sample_2.csv")
BC_inlet_IC_1 = load_CSV_data("BCs/BC_inlet_sample_2.csv")

# ================================
# ================================

BC_outlet_IC_2 = np.genfromtxt(os.path.join(config.DATA_PATH, "BCs", "BC_outlet_IC_2.csv"))
BC_inlet_IC_2 = np.genfromtxt(os.path.join(config.DATA_PATH, "BCs", "BC_inlet_IC_2.csv"))

# ================================
# ================================

# Goes with true parameters: z, rho_j, u, w = [170, 450, 3, 10]
# Goes with data_array: "artificial_data_array_flow_prior_BC_poisson.csv"
BC_inlet_sample_prior = np.genfromtxt(os.path.join(config.DATA_PATH, "BCs", "BC_inlet_sample_prior.csv"))
BC_outlet_sample_prior = np.genfromtxt(os.path.join(config.DATA_PATH, "BCs", "BC_outlet_sample_prior.csv"))

# ================================
# ================================

# Goes with true parameters:  z, rho_j, u, w = [160, 500, 3, 10]
BC_inlet_sample_prior_2 = np.genfromtxt(os.path.join(config.DATA_PATH, "BCs", "BC_inlet_sample_prior_2.csv"))
BC_outlet_sample_prior_2 = np.genfromtxt(os.path.join(config.DATA_PATH, "BCs", "BC_outlet_sample_prior_2.csv"))

# ================================
# ================================

# Only 20 time intervals
# Goes with true parameters:  z, rho_j, u, w = [160, 400, 3, 10]
BC_inlet_sample_prior_short = np.genfromtxt(os.path.join(config.DATA_PATH, "BCs", "BC_inlet_sample_prior_short.csv"))
BC_outlet_sample_prior_short = np.genfromtxt(os.path.join(config.DATA_PATH, "BCs", "BC_outlet_sample_prior_short.csv"))


# ================================
# ================================

# 3 BCs from 3 chains running on artifical data: 'artificial_data_array_flow_prior_BC_poisson.csv'
BC_inlet_art_1 = np.genfromtxt(os.path.join(config.DATA_PATH, "BCs", "3_BCs/BC_inlet_1_at_5000.csv"))
BC_outlet_art_1 = np.genfromtxt(os.path.join(config.DATA_PATH, "BCs", "3_BCs/BC_outlet_1_at_5000.csv"))

BC_inlet_art_2 = np.genfromtxt(os.path.join(config.DATA_PATH, "BCs", "3_BCs/BC_inlet_2_at_5000.csv"))
BC_outlet_art_2 = np.genfromtxt(os.path.join(config.DATA_PATH, "BCs", "3_BCs/BC_outlet_2_at_5000.csv"))

BC_inlet_art_3 = np.genfromtxt(os.path.join(config.DATA_PATH, "BCs", "3_BCs/BC_inlet_3_at_5000.csv"))
BC_outlet_art_3 = np.genfromtxt(os.path.join(config.DATA_PATH, "BCs", "3_BCs/BC_outlet_3_at_5000.csv"))

# ================================
# ================================

# BCs for testing bimodality
# goes with FD parameters: 160, 500, 3.5, 10
BC_inlet_bimodality = np.genfromtxt(os.path.join(config.DATA_PATH, "BCs", "BC_inlet_bimodality.csv"))
BC_outlet_bimodality = np.genfromtxt(os.path.join(config.DATA_PATH, "BCs", "BC_outlet_bimodality.csv"))

# ================================
# ================================
# 3 BCs from 3 chains running on real data: 'data_array_flow_70108_longer.csv'
BC_inlet_real_data_1 = np.genfromtxt(os.path.join(config.DATA_PATH, "BCs", "3_BCs_real_data_May/BC_inlet_1_at_15000.csv"))
BC_outlet_real_data_1 = np.genfromtxt(os.path.join(config.DATA_PATH, "BCs", "3_BCs_real_data_May/BC_outlet_1_at_15000.csv"))

BC_inlet_real_data_2 = np.genfromtxt(os.path.join(config.DATA_PATH, "BCs", "3_BCs_real_data_May/BC_inlet_2_at_15000.csv"))
BC_outlet_real_data_2 = np.genfromtxt(os.path.join(config.DATA_PATH, "BCs", "3_BCs_real_data_May/BC_outlet_2_at_15000.csv"))

BC_inlet_real_data_3 = np.genfromtxt(os.path.join(config.DATA_PATH, "BCs", "3_BCs_real_data_May/BC_inlet_2_at_15000.csv"))
BC_outlet_real_data_3 = np.genfromtxt(os.path.join(config.DATA_PATH, "BCs", "3_BCs_real_data_May/BC_outlet_2_at_15000.csv"))

# ================================
# ================================
# real data: 70129 - AM peak
# shorter: go with `data_array_short_70129_AM_flow.csv`:
# 70:180 onwards cuts FF and CF a bit to make it computationally more manageable
BC_inlet_70129 = np.genfromtxt(os.path.join(config.DATA_PATH, "BCs", "70129/BC_inlet_70129.csv"))[70:181]
BC_outlet_70129 = np.genfromtxt(os.path.join(config.DATA_PATH, "BCs", "70129/BC_outlet_70129.csv"))[70:181]

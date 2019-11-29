import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
sns.set_style("darkgrid")

from BIP_LWR.lwr.lwr_solver import LWR_Solver
from BIP_LWR.tools.util import FD_neg_power
from BIP_LWR.tools.BC_ICs import BC_inlet_real_data_3, BC_outlet_real_data_3
from BIP_LWR.tools.vis.vis_util import plot_LWR_xt_on_ax
from BIP_LWR.traffic_data.chain_data.DelCast_DS1_FDBC.FD_DelCast_DS1_FDBC import FD_delcast_ds1_FDBC

# define config info for the LWR solver
config_dict = {'my_analysis_dir': '2018/June3_2018-DS1_del_Cast-rawBCs',
                'run_num': 1,
                'data_array_dict':
                        {'flow': 'data_array_70108_flow_49t.csv',
                        'density': 'data_array_70108_density_49t.csv'},
                # 'ratio_times_BCs': 40,
				# 'step_save': 31,
                      }
LWR = LWR_Solver(config_dict=config_dict)

# ========================
# plot Fundamental Diagram along with flow-density data
FD_params = {'rho_j': 436.54, 'u': 3.2151, 'w': 28, 'z': 168.5}
x_range = np.arange(0,FD_params['rho_j'])
FD_array = FD_neg_power(rho=x_range, **FD_params)

fig1 = plt.figure(figsize=(10,8))
ax = fig1.add_subplot(1,1,1)
ax.plot(x_range, FD_array, alpha=0.7, c='r')
ax.scatter(LWR.df_FD_data.density, LWR.df_FD_data.flow, alpha=0.5)
ax.set_xlabel("Density", size=19)
ax.set_ylabel("Flow", size=19)
ax.set_title("Fundamental diagram with flow-density data", size=18)


# ========================
# compute Poisson loss for these FD parameters and boundary conditions
import copy
FD_params = copy.deepcopy(FD_delcast_ds1_FDBC)
FD_params.update({'solver': 'lwr_del_Cast'})# , 'BC_outlet': BC_outlet_real_data_3, 'BC_inlet': BC_inlet_real_data_3})


print("Poisson loss: {:.3f}".format(LWR.loss_function(**FD_params)))

# ========================


# ========================
# Show image of real data to visually compare the model to
density_image = 'BIP_LWR/traffic_data/images/M25-70108_DS1_density.png'
# flow_image = 'BIP_LWR/traffic_data/images/M25_Data_flow_longer-good_orientation.png'

fig_true_data, (ax1_td, ax2_td) = plt.subplots(2, figsize=(7, 7))
ax1_td.imshow(mpimg.imread(density_image))
ax1_td.set_title("Density data", size=15)
# plot output of LWR
lwr_ax = plot_LWR_xt_on_ax(FD=FD_params, data_variable='density', ax=ax2_td, fig=fig_true_data, config_dict=config_dict)
# ax2_td.imshow(mpimg.imread(flow_image))
# ax2_td.set_title("Flow data", size=15)
plt.show()

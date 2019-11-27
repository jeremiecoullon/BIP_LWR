# -*- coding: utf-8 -*-


def check_sampler_param_order(ICs, FD_type):
	"""
	Checks whether the parameters (FD and BCs) are in the correct order

	Parameters 
	----------
	ICs: dict
		Dictionary of initial conditions
	FD_type: str 	
		Either 'exp' or 'del_Cast'
	"""
	if FD_type == 'exp':
		list1 = ['alpha', 'beta']
		list2 = ['alpha', 'beta', 'BC_outlet', 'BC_inlet']
		list3 = ['alpha', 'beta', 'BC_outlet', 'BC_inlet', 'beta_temp_idx']
	elif FD_type == 'del_Cast':
		list1 = ['z', 'rho_j', 'u', 'w']
		list2 = ['z', 'rho_j', 'u', 'w', 'BC_outlet', 'BC_inlet']
		list3 = ['z', 'rho_j', 'u', 'w', 'BC_outlet', 'BC_inlet', 'beta_temp_idx']
	if list(ICs.keys()) != list1 and list(ICs.keys()) != list2 and list(ICs.keys()) != list3:
	            raise ValueError("The order of parameters should be {} (possibly with 'beta_temp_idx' at the end)".format(list2))
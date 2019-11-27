# -*- coding: utf-8 -*-


import numpy as np



def choose_pairs_four(PT_width):
	# manually do the case of width = 1
	if PT_width == 1:
		lecoin = np.random.binomial(n=1, p=0.5)
		if lecoin == 0:
			return [[0,1], [2,3]]
		elif lecoin == 1:
			return [[1,2]]
	else:
		pass

	half_width = PT_width//2
	# choose `half_width+1` number of chains from the untempered chains
	r_1 = list(np.random.choice((range(0, PT_width)), replace=False, size=half_width+1))

	# sampled all chains from 2nd temperature without replacement
	r_2 = list(np.random.choice((range(PT_width, 2*PT_width)), replace=False, size=PT_width))
	list_1 = []
	for x,y in zip(r_1, r_2[0:half_width]):
		list_1.append([x,y])

	# 3rd temperature
	r_3 = list(np.random.choice((range(2*PT_width, 3*PT_width)), replace=False, size=PT_width))
	list_2 = []
	for x,y in zip(r_2[half_width:], r_3[0:half_width]):
		list_2.append([x,y])

	# 4th temperature
	r_4 = list(np.random.choice((range(3*PT_width, 4*PT_width)), replace=False, size=half_width+1))
	list_3 = []
	for x,y in zip(r_3[half_width:], r_4):
		list_3.append([x,y])

	if PT_width % 2 == 1:
		extra_list = [[r_1[half_width], r_2[-1]]]
		return list_1 + list_2 + list_3 + extra_list
	else:
		return list_1 + list_2 + list_3

def choose_pairs_three(PT_width):
	"""
	Swapping schedule for 3 temperatures.
	"""
	# manually do the case of width = 1
	if PT_width == 1:
		lecoin = np.random.binomial(n=1, p=0.5)
		if lecoin == 0:
			return [[0,1], [2,3]]
		elif lecoin == 1:
			return [[1,2]]
	else:
		pass

	half_width = PT_width//2
	# choose `half_width+1` number of chains from the untempered chains
	r_1 = list(np.random.choice((range(0, PT_width)), replace=False, size=half_width+1))

	# sampled all chains from 2nd temperature without replacement
	r_2 = list(np.random.choice((range(PT_width, 2*PT_width)), replace=False, size=PT_width))
	list_1 = []
	for x,y in zip(r_1, r_2[0:half_width]):
		list_1.append([x,y])

	# 3rd temperature
	r_3 = list(np.random.choice((range(2*PT_width, 3*PT_width)), replace=False, size=half_width+1))
	list_2 = []
	for x,y in zip(r_2[half_width:], r_3):
		list_2.append([x,y])

	return list_1 + list_2

def choose_pairs_two(PT_width):
	"""
	Create pairs for depth=2

	Returns: list of lists
	"""
	r_1 = list(np.random.choice((range(0, PT_width)), replace=False, size=PT_width))
	# sampled all chains from 2nd temperature without replacement
	r_2 = list(np.random.choice((range(PT_width, 2*PT_width)), replace=False, size=PT_width))
	list_1 = []
	for x,y in zip(r_1, r_2):
		list_1.append([x,y])
	return list_1

def choose_pairs(PT_width, depth):
	"""
	Choose pairs to do swaps for Population PT Sampler
	"""
	if depth == 2:
		return choose_pairs_two(PT_width=PT_width)
	elif depth == 4:
		return choose_pairs_four(PT_width=PT_width)
	elif depth == 3:
		return choose_pairs_three(PT_width=PT_width)
	else:
		raise ValueError("Only works with depth 2, 3, and 4")

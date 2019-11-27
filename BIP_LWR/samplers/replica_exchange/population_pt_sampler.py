#!/usr/bin/env python

import numpy as np

from mpi4py import MPI
from BIP_LWR.tools import util
from BIP_LWR.samplers.mhsampler import MHSampler
from BIP_LWR.moves.gaussian import GaussianMove
from .util import choose_pairs

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()



def get_temp_idx(rank, PT_width):
	"""
	Get index for MCMC chain given rank and PT_width
	idx=0 is the untempered chain
	"""
	return (rank-1)//M + 1





class PopulationPTSampler:
	"""
	Population Parallel Tempering Sampler: the target samplers all share hot chains.

	Parameters
	----------
	dict_samplers: dict
		Dictionary of functions that instantiate a sampler (these take no argument)
		format: {0: samplerfun0, 1: samplerfun1, 2: samplerfun2,}
	within_temp_iter: int
		Number of within temperatures moves to do
	PT_width: int
		Number of chains to have for each temperature (except for the untempered chain)
	"""
	def __init__(self, dict_samplers, within_temp_iter, PT_width):
		self.rank = rank
		self.PT_width = PT_width
		self.within_temp_iter = within_temp_iter

		if self.rank in range(0, PT_width):
			self.my_mcmc = dict_samplers[0]()
		elif self.rank in range(PT_width, 2*PT_width):
			self.my_mcmc = dict_samplers[1]()
		elif self.rank in range(2*PT_width, 3*PT_width):
			self.my_mcmc = dict_samplers[2]()
		elif self.rank in range(3*PT_width, 4*PT_width):
			self.my_mcmc = dict_samplers[3]()
		else:
			raise ValueError("Just defined 4 temperatures for now")

		self.my_mcmc.save_at_end = False
		# only the untempered chain saves
		# if self.rank != 0:
		# 	self.my_mcmc.config.save_chain = False
		# set process number to mcmc so it saves it under a specific filename
		self.my_mcmc.backend.proc_num = self.rank

		# number of temperatures
		self.num_temps = len(dict_samplers)
		# set 2 for now
		# self.num_temps = 2
		if size != (self.num_temps*PT_width):
			long_dash = "\n\n===================================================================================\n\n"
			raise ValueError(long_dash + "Error in number of processes. Expected {}; got {}".format((self.num_temps*PT_width), size) + long_dash)

	def check_swap(self, list_pairs):
		"""
		Check whether the current process should do a temperature swap move

		Parameters
		----------
		list_pairs: list
			List of lists: pairs that will do a swap move

		Returns
		-------
		do_swap: Bool
			Whether or not to swap
		temp_pair: list or None
			If swap: list of 2 temps. Else: None
		"""
		list_bools = [self.rank in e for e in list_pairs]
		if True in list_bools:
			do_swap = True
			temp_pair = list_pairs[list_bools.index(True)]
		else:
			do_swap = False
			temp_pair = None
		return do_swap, temp_pair

	def temp_swap_move(self, temp_pair):
		"""
		Do a temperature swap move.
		"""
		# get rank of the other process to do a swap with
		idx_rank = temp_pair.index(self.rank)
		idx_other_rank = (idx_rank+1)%2
		other_rank = temp_pair[idx_other_rank]

		# exchange current_samples
		# Add barrier to avoid deadlock when sending large arrays
		if self.rank == min(temp_pair):
			comm.send(self.my_mcmc.backend.current_samples, dest=other_rank)
			comm.Barrier()
			current_samples_other = comm.recv(source=other_rank)
		elif self.rank == max(temp_pair):
			current_samples_other = comm.recv(source=other_rank)
			comm.Barrier()
			comm.send(self.my_mcmc.backend.current_samples, dest=other_rank)

		loss_new = self.my_mcmc.log_posterior(**current_samples_other)
		if self.rank == max(temp_pair):
			comm.send({'loss_current': self.my_mcmc.backend.loss_current, 'loss_new': loss_new},
					dest=other_rank)
			accepted = comm.recv(source=other_rank)
		elif self.rank == min(temp_pair):
			dict_loss_other = comm.recv(source=other_rank)
			# accept-reject
			alpha = (loss_new + dict_loss_other['loss_new']) - (self.my_mcmc.backend.loss_current + dict_loss_other['loss_current'])
			exp_sample = - np.random.exponential()
			if alpha > exp_sample:
				accepted = True
			else:
				accepted = False
			comm.send(accepted, dest=other_rank)

		# update backend
		if accepted == True:
			self.my_mcmc.backend.save_step(new_samples=current_samples_other, loss_new=loss_new, accepted=accepted)
			# try:
			self.my_mcmc.backend.update_sample_params("PT_beta_a")
			# except AttributeError:
			# 		pass
		else:
			self.my_mcmc.backend.save_step(new_samples=self.my_mcmc.backend.current_samples,
				loss_new=self.my_mcmc.backend.loss_current, accepted=accepted)
			# try:
			self.my_mcmc.backend.update_sample_params("PT_beta_r")
			# except AttributeError:
			# 		pass
		try:
			self.my_mcmc.backend.update_current_section(self.my_mcmc.move.BC_move.current_section)
		except AttributeError:
			pass

	@util.time_it
	def run(self, n_iter, print_rate):
		"Run pt sampler"
		for iter_num in range(n_iter):
			if iter_num%print_rate==0:
				print("PT sampler: starting iteration {}/{}".format(iter_num, n_iter))
			self.my_mcmc.run(self.within_temp_iter, self.within_temp_iter+1)


			if rank != 0:
				list_pairs = None
			elif rank == 0:
				list_pairs = choose_pairs(PT_width=self.PT_width, depth=self.num_temps)

			list_pairs = comm.bcast(list_pairs, root=0)
			# print("On rank {}. list_pairs={}".format(self.rank, list_pairs))

			do_swap, temp_pair = self.check_swap(list_pairs=list_pairs)
			if do_swap == True:
				# print("Rank {} doing a swap.".format(self.rank))
				self.temp_swap_move(temp_pair=temp_pair)
			else:
				self.my_mcmc.backend.save_step(new_samples=self.my_mcmc.backend.current_samples,
					loss_new=self.my_mcmc.backend.loss_current, accepted=False)
				# try:
				self.my_mcmc.backend.update_sample_params("PT_beta_not_chosen")
				# except AttributeError:
				# 	pass
				try:
					self.my_mcmc.backend.update_current_section(self.my_mcmc.move.BC_move.current_section)
				except AttributeError:
					pass
				# need this barrier for the chains not doing a temperature swap
				comm.Barrier()
		if self.my_mcmc.config.save_chain == True:
			self.my_mcmc.save_to_file(iter_num=0, iter_step=1)

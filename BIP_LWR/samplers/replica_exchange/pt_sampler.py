#!/usr/bin/env python

import numpy as np

from mpi4py import MPI
from BIP_LWR.tools import util
from BIP_LWR.samplers.mhsampler import MHSampler
from BIP_LWR.moves.gaussian import GaussianMove

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


class PTSampler:
	"""
	Parallel Tempering Sampler

	Parameters
	----------
	dict_samplers: dict
		Dictionary of functions that instantiate a sampler (these take no argument)
		format: {0: samplerfun0, 1: samplerfun1, 2: samplerfun2,}
	within_temp_iter: int
		Number of within temperatures moves to do
	"""
	def __init__(self, dict_samplers, within_temp_iter):
		self.rank = rank
		self.within_temp_iter = within_temp_iter
		self.my_mcmc = dict_samplers[self.rank]()
		self.my_mcmc.save_at_end = False
		# set process number to mcmc so it saves it under a specific filename
		self.my_mcmc.backend.proc_num = self.rank

		# number of temperatures
		self.num_temps = len(dict_samplers)
		if self.num_temps != size:
			long_dash = "\n\n===================================================================================\n\n"
			raise ValueError(long_dash + "Must have number of processes equal to the number of temperatures. Expected {}; got {}".format(self.num_temps, size) + long_dash)

	def check_swap(self, temp_pair):
		"""
		Check whether the current process should do a temperature swap move
		"""
		if self.rank in temp_pair:
			do_swap = True
		else:
			do_swap = False
		return do_swap

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
			self.my_mcmc.backend.update_sample_params("PT_beta_a")
		else:
			self.my_mcmc.backend.save_step(new_samples=self.my_mcmc.backend.current_samples,
				loss_new=self.my_mcmc.backend.loss_current, accepted=accepted)
			self.my_mcmc.backend.update_sample_params("PT_beta_r")
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

			# schedule for choose pairs of temperatures
			iter_mod = iter_num%(self.num_temps-1)
			temp_pair = [iter_mod, iter_mod+1]
			do_swap = self.check_swap(temp_pair=temp_pair)
			if do_swap == True:
				self.temp_swap_move(temp_pair=temp_pair)
			else:
				self.my_mcmc.backend.save_step(new_samples=self.my_mcmc.backend.current_samples,
					loss_new=self.my_mcmc.backend.loss_current, accepted=False)
				self.my_mcmc.backend.update_sample_params("PT_beta_not_chosen")
				try:
					self.my_mcmc.backend.update_current_section(self.my_mcmc.move.BC_move.current_section)
				except AttributeError:
					pass
				# need this barrier for the chains not doing a temperature swap
				comm.Barrier()
		if self.my_mcmc.config.save_chain == True:
			self.my_mcmc.save_to_file(iter_num=0, iter_step=1)

# BIP_LWR

Bayesian inverse problem for LWR


## installation

- `virtualenv venv; source venv/bin/activate`
- `pip install clawpack` (for some reason this package sometimes needs to be installed separately)
- `pip install -r requirements.txt`
- `make` (for the fortran solvers)
- `make test`

Also need:
- `gfortran` for the PDE solver
- `awscli` to save chains to S3 (for `lwr_sampler` if `upload_to_S3=True`)


## Demos

1. Run demo: `python -m BIP_LWR.demos.demo_mh_gaussian`.
  - Samples a 2D Gaussian using MH
  - plots the trace plots and kdeplots:


2. Run demo: `python -m BIP_LWR.demos.demo_lwr_solver`
  - plots FD along with flow-density data
  - solves LWR with del Castillo using some parameters (and BCs sampled from the joint posterior)
  - prints poisson loss
  - plots LWR output in x-t plane compared to data


## Paper results

### Run samplers

Result 1 below (the notebook) runs the sampler and creates the figures found in the paper. Result 2-5 only run the MCMC sampler and saves the output in hdf5 files.

If `upload_to_S3 = False` (in the relevant scripts) then the hdf5 file is created in the directory `../Analysis/`.

To save the output to S3 (for long runs):
- change the bucket name in `BIP_LWR.tools.util.upload_chain` and `BIP_LWR.tools.util.download_chain`
- Add your AWS credentials using `aws configure` (see [documentation](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html))


1. Direct fit (section 3.2 in the paper): Run the notebook `BIP_LWR/demos/FD_direct_fit_MCMC-delCast.ipynb`.
2. FD only results (section 3.3): `./MCMC_FDonly.py`.
3. BC only; PT sampler (section 4.2.2): `mpiexec -n 3 ./MCMC_BConly_PT.py`
4. BC only; Population PT (section 4.3): `mpiexec -n 15 ./MCMC_BConly_PopulationPT.py`
5. FD & BC; Population PT (section 5.1): `mpiexec -n 16 ./MCMC_FDBC_PopulationPT.py`

### Analyse results

To analyse results from runs in the section above, run the notebooks in `BIP_LWR/demos/`. There is a notebook for each of the runs 2-5 above.


## Other

#### Settings

- Output saved in a pandas DataFrame in a hdf5 file (in `Analysis/FD_only_sampling/` for example). The file name is based on the initial conditions. The directory name is set manually (for example in `MCMC_FDonly.py` this is set at line 45: `'my_analysis_dir': 'FD_only_sampling'`).
- Can also save the output to a S3 bucket: change the bucket name in `BIP_LWR.tools.util.upload_chain` and `BIP_LWR.tools.util.download_chain`
- Each demo is set to run for a few iterations. For BC and (FD,BC) sampling, you need to run the chain for at least around 200K iterations to get enough posterior samples (This should take around 3 days)

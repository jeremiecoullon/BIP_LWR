# BIP_LWR

Bayesian inverse problem for LWR


## installation

- `virtualenv -p python3 venv; source venv/bin/activate`
- `pip install -r requirements.txt`. Note that clawpack (the PDE solver) takes a while to install.
- `make` (for the fortran solvers)
- `make test`

Also need:
- `gfortran` for the PDE solver
- `awscli` to save chains to S3 (for `lwr_sampler` if `upload_to_S3=True`)


## Demos

### Simple Metropolis Hasting example

Samples a 2D Gaussian using MH, and plot trace plots and kdeplots:

Run demo: `python -m BIP_LWR.experiments.demo_mh_gaussian`



### Solve LWR

You can read an overview of the LWR model [here](BIP_LWR/docs/lwr_overview.md)

Run demo: `python -m BIP_LWR.experiments.demo_lwr_solver`

- plots FD along with flow-density data
- solves LWR with del Castillo using some parameters (and BCs sampled from the joint posterior)
- prints poisson loss
- plots LWR output in x-t plane compared to data


### Run LWR MCMC

Run demo: `python -m BIP_LWR.experiments.demo_lwr_mcmc`

#### Sampler

Samples parameters in Del Castilo's FD along with boundary conditions (inlet and outlet).
MCMC moves:
1. Sample from $\pi(FD| BC)$ using Metropolis Hastings (random walk)
2. Sample from $pi(FD, BC)$ using a joint proposal that aims to keep flow constant.
3. Sample from $pi(BC | FD)$ using Gibbs blocks.

Can modify the move probabilities (currently: `move_probs = [0.1, 0.1, 0.8]`)

#### Settings

- Output saved in a pandas DataFrame in a hdf5 file (in "Analysis/2018/Demo_del_Castillo_MCMC/"). The file name is based on the initial conditions. The directory name is set manually (here it's "Demo_del_Castillo_MCMC/")
- Can save the output to a S3 bucket: change the bucket name in `BIP_LWR.tools.util.upload_chain` and `BIP_LWR.tools.util.download_chain`
- Demo is set to run for 2 iterations. Need to run for at least 10,000 iterations to get a idea of what the posterior looks like (This should take around 12 hours for 3 chains running simultaneously)

#### To diagnose chains


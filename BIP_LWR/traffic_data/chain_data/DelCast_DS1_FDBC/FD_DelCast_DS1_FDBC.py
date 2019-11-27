#!/usr/bin/env python

import numpy as np
from collections import OrderedDict
from BIP_LWR.tools.util import load_CSV_data


# Sample from Feb4; run_2; S3 (chain 2). Del Cast, DS1, with beta_temp = 0.44, and joint move (with BC shift)
BC_outlet = load_CSV_data(path='chain_data/DelCast_DS1_FDBC/DelCastDS1_FDBC_outlet.csv')
BC_inlet = load_CSV_data(path='chain_data/DelCast_DS1_FDBC/DelCastDS1_FDBC_inlet.csv')


FD_delcast_ds1_FDBC = OrderedDict([('z', 174.9521288),
             ('rho_j', 377.68307814),
             ('u', 3.76935867),
             ('w', 0.2118941),
             ('BC_outlet', BC_outlet),
             ('BC_inlet', BC_inlet),
                     ])
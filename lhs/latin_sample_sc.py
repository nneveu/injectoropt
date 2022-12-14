# """
# Runs libEnsemble with Latin hypercube sampling on a simple 1D problem
#
# Execute via one of the following commands (e.g. 3 workers):
#    mpiexec -np 4 python3 test_1d_sampling.py
#    python3 test_1d_sampling.py --nworkers 3 --comms local
#    python3 test_1d_sampling.py --nworkers 3 --comms tcp
#
# The number of concurrent evaluations of the objective function will be 4-1=3.
# """

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: mpi local tcp
# TESTSUITE_NPROCS: 2 4

import mpi4py
mpi4py.rc.recv_mprobe = False
from mpi4py import MPI

import os, sys, glob
import numpy as np
sys.path.append('/gpfs/slac/staas/fs1/g/accelerator_modeling/nneveu/injectoropt/')

# Import libEnsemble items for this test
from libensemble.libE import libE
from libensemble.gen_funcs.sampling import latin_hypercube_sample as gen_f
from libensemble.tools import parse_args, save_libE_output, add_unique_random_streams
from libensemble.message_numbers import WORKER_DONE, WORKER_KILL, TASK_FAILED, WORKER_KILL_ON_TIMEOUT
from libensemble.executors.mpi_executor import MPIExecutor

from libensemble import logger
logger.set_level('DEBUG')
nworkers, is_master, libE_specs, _ = parse_args()

TOP_DIR = '/gpfs/slac/staas/fs1/g/accelerator_modeling/nneveu/injectoropt/sc_files/'
#Setting up the simulation enviornment
FMAP_DIR = TOP_DIR + 'fieldmaps'

libE_specs = {"nworkers": nworkers, "comms": "local"}
libE_specs['save_every_k_gens'] = 100 
libE_specs['save_every_k_sims'] = 1
libE_specs['sim_dirs_make'] = True
libE_specs['sim_dir_symlink_files'] = [ f for f in glob.glob(FMAP_DIR+'/*.txt')]

# Create executor and register sim to it
exctr = MPIExecutor() 

#User sim_f
from libeopal import opal_sample
import opt_config

# Register simulation executable with executor
sim_app = '/gpfs/slac/staas/fs1/g/accelerator_modeling/nneveu/software/OPAL/opal_mpich/bin/opal'
exctr.register_app(full_path=sim_app, calc_type='sim')


STAT_NAMES = ['t', 's','numParticles','charge','energy','rms_x', 'rms_y', 'rms_s', \
              'rms_px', 'rms_py', 'rms_ps', 'emit_x', 'emit_y', 'emit_s', 'dE']#, 'mean_x', \
              #'mean_y', 'mean_s', 'ref_x', 'ref_y', 'ref_z', 'ref_px', 'ref_py', 'ref_pz', \
              #'max_x', 'max_y', 'max_s', 'xpx', 'ypy', 'zpz', 'Dx', 'DDx', 'Dy', 'DDy', \
              #'Bx_ref', 'By_ref', 'Bz_ref', 'Ex_ref', 'Ey_ref', 'Ez_ref', 'dE', 'dt', 'partsOutside']

# Keys related to data in OPAL sim and objectives
key_dict = {'data_keys': STAT_NAMES,
            'dvar_keys': list(opt_config.xbounds['name'][:]),
            'objective_keys':opt_config.objscale['name']}

num_objs = len(opt_config.objscale['name'])
# State the sim_f, its arguments, output, and parameters (and their sizes)
sim_specs = {'sim_f': opal_sample,         # sim_f, imported above
             'in': ['x'],                 # Name of input for sim_f
             # Name, type of output from sim_f, f is the function being minimized
             'out': [('f', float, num_objs)] +  [(key,float) for key in key_dict['data_keys']] + [(key+'_long',float, 2500) for key in key_dict['data_keys']],
             'user': {'key_dict': key_dict,
                      'basefile_name': 'sc_inj_C1',
                      'input_files_path':TOP_DIR,
                      'distgen_file':TOP_DIR+'tgauss.yaml',
                      'zstop': opt_config.zstop,
                      'penalty_scale':opt_config.penalty,
                      'xscales':opt_config.xscale,
                      'objective_scales':opt_config.objscale,
                      'cores': opt_config.cores,
                      'sim_particles': opt_config.simpart,
                      'sim_kill_minutes': opt_config.simkill,
			
                      }
             }
#sim_specs = {'sim_f': sim_f, 'in': ['x'], 'out': [('f', float)]}

gen_specs = {'gen_f': gen_f,
             'out': [('x', float, len(opt_config.xbounds['name']))],
             'user': {'gen_batch_size': 10,
                      'lb': opt_config.xbounds['lb'],
                      'ub': opt_config.xbounds['ub'],
                      }
             }

persis_info = add_unique_random_streams({}, nworkers + 1)

exit_criteria = {'sim_max': 10}

# Perform the run
H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info,
                            libE_specs=libE_specs)

if is_master:
    assert len(H) >= 999 
    print("\nlibEnsemble with random sampling has generated enough points")
    save_libE_output(H, persis_info, __file__, nworkers)

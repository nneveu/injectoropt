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

# Import libEnsemble items for this test
from libensemble.libE import libE
#from libensemble.sim_funcs.one_d_func import one_d_example as sim_f
from libensemble.gen_funcs.sampling import latin_hypercube_sample as gen_f
from libensemble.tools import parse_args, save_libE_output, add_unique_random_streams
from libensemble.message_numbers import WORKER_DONE, WORKER_KILL, TASK_FAILED, WORKER_KILL_ON_TIMEOUT
from libensemble.executors.mpi_executor import MPIExecutor

from libensemble import libE_logger
libE_logger.set_level('DEBUG')
nworkers, is_master, libE_specs, _ = parse_args()
libE_specs['save_every_k_gens'] = 100 
libE_specs['save_every_k_sims'] = 1
libE_specs['ensemble_dir_path'] = 'ensemble'
# Create executor and register sim to it
exctr = MPIExecutor()  # Use auto_resources=False to oversubscribe

#User sim_f
from libeopal import opal_sample
import opt_config

TOP_DIR = '/gpfs/slac/staas/fs1/g/accelerator_modeling/nneveu/emittance_minimization/code/slac/'
files   = 'paper_test/'
#Setting up the simulation enviornment
FMAP_DIR = TOP_DIR + 'fieldmaps'

# Register simulation executable with executor
sim_app = '/gpfs/slac/staas/fs1/g/accelerator_modeling/nneveu/software/OPAL/opal_mpich/bin/opal'
exctr.register_calc(full_path=sim_app, calc_type='sim')

libE_specs['comms'] = 'local'
libE_specs['sim_dir_symlink_files'] = [ f for f in glob.glob(FMAP_DIR+'/*.txt')]
libE_specs['sim_dirs_make'] = True



STAT_NAMES = ['t', 's','numParticles','charge','energy','rms_x', 'rms_y', 'rms_s', \
              'rms_px', 'rms_py', 'rms_ps', 'emit_x', 'emit_y', 'emit_s', 'dE']#, 'mean_x', \
              #'mean_y', 'mean_s', 'ref_x', 'ref_y', 'ref_z', 'ref_px', 'ref_py', 'ref_pz', \
              #'max_x', 'max_y', 'max_s', 'xpx', 'ypy', 'zpz', 'Dx', 'DDx', 'Dy', 'DDy', \
              #'Bx_ref', 'By_ref', 'Bz_ref', 'Ex_ref', 'Ey_ref', 'Ez_ref', 'dE', 'dt', 'partsOutside']

objscale =  np.array([
        ('emit_x',0.3e-6,6e-6),
        ('rms_s',0.5e-3,4e-3),
        ('dE', 0.0, 1.0),
        ],dtype=[('name', 'U10'), ('lb', 'f4'), ('ub', 'f4')])

xscale   = np.array([('PHBUN', 1.0), ('GBUN',0.1), \
                     ('SF1', 1e-3), ('SF2', 1e-3), \
                    ('PHCM1', 1.0), ('GCM1', 1.0), \
                    ('PHCM2', 1.0), ('GCM2', 1.0),\
                    ('PHCM3', 1.0), ('GCM3', 1.0), \
                    ('PHCM4', 1.0), ('GCM4', 1.0),\
                   ],dtype=[('name', 'U10'), ('scale', 'f4')])

xbounds  = np.array([('PHBUN', -100.0, -10.0), ('GBUN', 10.0, 18.0), \
                    ('SF1', 20.0, 70.0), ('SF2', 20.0, 70.0), \
                    ('PHCM1', -40.0, 40.0), ('GCM1', 0.0, 32.0), \
                    ('PHCM2', -40.0, 40.0), ('GCM2', 0.0, 32.0), \
                    ('PHCM3', -40.0, 40.0), ('GCM3', 0.0, 32.0), \
                    ('PHCM4', -40.0, 40.0), ('GCM4', 0.0, 32.0), \
                   ],dtype=[('name', 'U10'), ('lb', 'f4'), ('ub', 'f4')])

# Keys related to data in OPAL sim and objectives
key_dict = {'data_keys': STAT_NAMES,
            'dvar_keys': list(xbounds['name'][:]),
            'objective_keys':objscale['name']}

num_objs = len(objscale['name'])
# State the sim_f, its arguments, output, and parameters (and their sizes)
sim_specs = {'sim_f': opal_sample,         # sim_f, imported above
             'in': ['x'],                 # Name of input for sim_f
             # Name, type of output from sim_f, f is the function being minimized
             'out': [('f', float, num_objs)] +  [(key,float) for key in key_dict['data_keys']] + [(key+'_long',float, 2500) for key in key_dict['data_keys']],
             'user': {'key_dict': key_dict,
                      'basefile_name': 'sc_inj_C1',
                      'input_files_path':TOP_DIR+files,
                      'distgen_file':TOP_DIR+files+'tgauss.yaml',
                      'zstop': 15.0,
                      'penalty_scale':opt_config.penalty,
                      'xscales':xscale,
                      'objective_scales':objscale,
                      'cores': 4,
                      'sim_particles': opt_config.simpart,
                      'sim_kill_minutes': 10,
			
                      }
             }
#sim_specs = {'sim_f': sim_f, 'in': ['x'], 'out': [('f', float)]}

gen_specs = {'gen_f': gen_f,
             'out': [('x', float, len(xbounds['name']))],
             'user': {'gen_batch_size': 10,
                      'lb': xbounds['lb'],
                      'ub': xbounds['ub'],
                      }
             }

persis_info = add_unique_random_streams({}, nworkers + 1)

exit_criteria = {'sim_max': 1000}

# Perform the run
H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info,
                            libE_specs=libE_specs)

if is_master:
    assert len(H) >= 999 
    print("\nlibEnsemble with random sampling has generated enough points")
    save_libE_output(H, persis_info, __file__, nworkers)

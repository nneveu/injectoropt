"""
Runs libEnsemble with APOSMM with the NLopt local optimizer.

Execute via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python3 test_persistent_aposmm_nlopt.py
   python3 test_persistent_aposmm_nlopt.py --nworkers 3 --comms local
   python3 test_persistent_aposmm_nlopt.py --nworkers 3 --comms tcp

When running with the above commands, the number of concurrent evaluations of
the objective function will be 2, as one of the three workers will be the
persistent generator.
"""

# Do not change these lines - they are parsed by run-tests.sh
# TESTSUITE_COMMS: local mpi tcp
# TESTSUITE_NPROCS: 3
# TESTSUITE_EXTRA: true

import sys, os, glob
import numpy as np
sys.path.append('/gpfs/slac/staas/fs1/g/accelerator_modeling/nneveu/emittance_minimization/code/slac/paper_test')

# Import libEnsemble items for this test
from libensemble.libE import libE
from math import gamma, pi, sqrt
#from libensemble.sim_funcs.six_hump_camel import six_hump_camel as sim_f
from libensemble.utils.timer import Timer
import libensemble.gen_funcs

libensemble.gen_funcs.rc.aposmm_optimizers = 'nlopt'
from libensemble.gen_funcs.persistent_aposmm import aposmm as gen_f

from libensemble.alloc_funcs.persistent_aposmm_alloc import persistent_aposmm_alloc as alloc_f
from libensemble.tools import parse_args, save_libE_output, add_unique_random_streams
#from libensemble.tests.regression_tests.support import six_hump_camel_minima as minima
from libensemble.message_numbers import WORKER_DONE, WORKER_KILL, TASK_FAILED, WORKER_KILL_ON_TIMEOUT
from libensemble.executors.mpi_executor import MPIExecutor
exctr = MPIExecutor()
timer = Timer()

from time import time
from libensemble import logger
logger.set_level('DEBUG')
from libeopal import opal_aposmm as sim_f
#import settings from opt_config file in paper_test directory
import opt_config
# Create executor and register sim to it
print(opt_config.sim_app)
exctr.register_app(full_path=opt_config.sim_app, calc_type='sim')

nworkers, is_manager, libE_specs, _ = parse_args()

if is_manager:
    start_time = time()

if nworkers < 2:
    sys.exit("Cannot run with a persistent worker if only one worker -- aborting...")

n = len(opt_config.xscale)
print('number of dimensions in x', n)
gen_out = [
    ('x', float, n),
    ('x_on_cube', float, n),
    ('sim_id', int),
    ('local_min', bool),
    ('local_pt', bool),
]

gen_specs = {
    'gen_f': gen_f,
    'persis_in': ['f'] + [n[0] for n in gen_out],
    'in': ['x','x_on_cube','f','returned','sim_id'],
    'out': gen_out,
    'user': {
        'initial_sample_size': 0,
        #'sample_points': 10 #np.round(minima, 1),
        'localopt_method': 'LN_BOBYQA',
        'rk_const': 0.5 * ((gamma(1 + (n / 2)) * 5) ** (1 / n)) / sqrt(pi),
        'xtol_abs': 1e-6,
        'ftol_abs': 1e-6,
        'dist_to_bound_multiple': 0.5,
        'max_active_runs': 3,
        'lb': opt_config.xbounds['lb'],
        'ub': opt_config.xbounds['ub'],
    },
}

exit_criteria = {'sim_max': 10}

num_objs = len(opt_config.objscale)
# Load previous sample
sample_size = 1000
f           = np.zeros((sample_size, num_objs))
sample      = np.load('/gpfs/slac/staas/fs1/g/accelerator_modeling/nneveu/emittance_minimization/code/slac/paper_test/sample/latin_sample_run5.npy')

# Initialize H0
H0_dtype = [
    ('x', float, n),
    ('x_on_cube', float, n),
    ('f', float),
    ('sim_id', int),
    ('returned', bool),
    ('given', bool),
    ('given_back', bool),
    ('emit_x', float),
    ('rms_s', float),
    ('dE',float),
]

H0           = np.zeros(sample_size, dtype=H0_dtype)
H0['x']      = sample['x']
H0['sim_id'] = range(sample_size)
H0[['given', 'given_back', 'returned']] = True
H0['x_on_cube'] = (H0['x']-gen_specs['user']['lb']) / (gen_specs['user']['ub']-gen_specs['user']['lb'])

emit        = sample[:sample_size]['emit_x']
zrms        = sample[:sample_size]['rms_s']
dE          = sample[:sample_size]['dE']

# Penatly check
num_penalty  = (opt_config.simpart - sample[:sample_size]['numParticles']) / opt_config.penalty
emit_penalty = np.where(emit<0.3e-6)
#print(emit_penalty)

# Calculate fitness values
raw_objs       = np.column_stack([emit, zrms, dE])
scaled_objs    = opt_config.scale_objs(raw_objs)
fvals          = scaled_objs + np.tile(num_penalty.reshape(1000,1), (1,3))
fvals_scaled   = opt_config.emit_scale(scaled_objs, fvals)
# "Load in" the points and their function values. 
#H0['f'] = fvals_scaled

# Persistent info between iterations
persis_info = add_unique_random_streams({}, nworkers + 1)
persis_info['next_to_give'] = 0 if H0 is None else len(H0)
persis_info['total_gen_calls'] = 0

libE_specs['sim_dir_symlink_files'] = [ f for f in glob.glob(opt_config.FMAP_DIR+'/*.txt')]
libE_specs['sim_dirs_make'] = True
libE_specs['save_every_k_sims'] = 1
libE_specs['save_every_k_gens'] = 1
libE_specs['ensemble_dir_path'] = 'ensemble'



# !!!!! LOOP WILL START HERE !!!!!!!!!!
w = round(1/3, 3)
weights = np.array([w])*3

 
sim_specs = {
    'sim_f': sim_f,
    'in': ['x'],
    #'out': [('f', float)],
    'out': [('f', float)]+ [(key,float) for key in opt_config.key_dict['data_keys']] + [(key+'_long',float, 2500) for key in opt_config.key_dict['data_keys']],
    'user': {'weights':w,
             'key_dict': opt_config.key_dict,
             'basefile_name': 'sc_inj_C1',
             'input_files_path':opt_config.TOP_DIR+opt_config.files,
             'zstop': opt_config.zstop,
             'penalty_scale': opt_config.penalty,
             'xscales':opt_config.xscale,
             'objective_scales':opt_config.objscale,
             'cores': opt_config.cores,
             'sim_particles':opt_config.simpart,
             'sim_kill_minutes': opt_config.simkill,
             } # end user specs
}# end sim specs

alloc_specs = {'alloc_f': alloc_f}

three_fs    = w*fvals_scaled
H0['f']     = np.sum(three_fs,axis=1) 
print('fvals_scaled', fvals_scaled[:3])
print('fvals*w', H0['f'][:3])



# Perform the run
H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info, alloc_specs, libE_specs=libE_specs,H0=H0)

if is_manager:
    print('[Manager]:', H[np.where(H['local_min'])]['x'])
    print('[Manager]: Time taken =', time() - start_time, flush=True)
    # Saving data to file
    script_name = 'w' + str(w)[0] +'_' + os.path.splitext(os.path.basename(__file__))[0]
    save_libE_output(H, persis_info, script_name, nworkers)

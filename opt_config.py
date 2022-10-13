import numpy as np

TOP_DIR = '/gpfs/slac/staas/fs1/g/accelerator_modeling/nneveu/emittance_minimization/code/slac/'
files   = 'paper_test/'
#Setting up the simulation enviornment
FMAP_DIR = TOP_DIR + 'fieldmaps'

# Register simulation executable with executor
sim_app = '/gpfs/slac/staas/fs1/g/accelerator_modeling/nneveu/software/OPAL/opal_mpich/bin/opal'


STAT_NAMES = ['t', 's','numParticles','charge','energy','rms_x', 'rms_y', 'rms_s', \
              'rms_px', 'rms_py', 'rms_ps', 'emit_x', 'emit_y', 'emit_s', 'dE']#, 'mean_x', \
              #'mean_y', 'mean_s', 'ref_x', 'ref_y', 'ref_z', 'ref_px', 'ref_py', 'ref_pz', \
              #'max_x', 'max_y', 'max_s', 'xpx', 'ypy', 'zpz', 'Dx', 'DDx', 'Dy', 'DDy', \
              #'Bx_ref', 'By_ref', 'Bz_ref', 'Ex_ref', 'Ey_ref', 'Ez_ref', 'dE', 'dt', 'partsOutside']

simkill = 10.0
simpart = 5e4
penalty = 20.0
cores   = 4
zstop   = 15.0
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
                    ('PHCM1', -40.0, 40.0), ('GCM1', 1.0e-5, 32.0), \
                    ('PHCM2', -40.0, 40.0), ('GCM2', 1.0e-5, 32.0), \
                    ('PHCM3', -40.0, 40.0), ('GCM3', 1.0e-5, 32.0), \
                    ('PHCM4', -40.0, 40.0), ('GCM4', 1.0e-5, 32.0), \
                   ],dtype=[('name', 'U10'), ('lb', 'f4'), ('ub', 'f4')])

# Keys related to data in OPAL sim and objectives
key_dict = {'data_keys': STAT_NAMES,
            'dvar_keys': list(xbounds['name'][:]),
            'objective_keys':objscale['name']}

num_objs = len(objscale['name'])

def scale_objs(raw_obj_vals):
    """Return scaled objectives"""
    scaled_obj_vals = np.zeros_like(raw_obj_vals)
    #print(scaled_obj_vals.shape)
    obj_names   = objscale['name']

    for row in range(0,raw_obj_vals.shape[0]):
        #print(raw_obj_vals.shape, raw_obj_vals.shape[0])
        # Using objscale['name'] ensures same order each time
        for i, key in enumerate(obj_names):
            obj_index      = np.where(objscale['name']==key)
            denominator    = objscale['ub'][obj_index][0] - objscale['lb'][obj_index][0]
            scaled_obj_vals[row, i] = (raw_obj_vals[row,i]-objscale['lb'][obj_index][0])/denominator
    return scaled_obj_vals


def emit_scale(objs_scaled, fvals):
    """penalize based on emittance value"""  
    emit_scaled  = objs_scaled[:,0] 
    index        = np.where(emit_scaled>0.3)
    emit_penalty = emit_scaled[index]-0.3 
    #print('scaled emit is', emit_scaled[index][:5])
    #print('emit penalty', emit_penalty.shape)
    emit_penalty = np.tile(emit_penalty.reshape(emit_penalty.shape[0],1), (1,3))
    #print('emit  penalty', emit_penalty[:3])
    #print('fvals', fvals[:3])
    fvals[index] = fvals[index] + emit_penalty
    return fvals

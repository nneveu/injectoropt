import numpy as np
import matplotlib.pyplot as plt
import glob, sys, h5py
import seaborn as sns

from astra import Astra, template_dir
import distgen, ssnl
from distgen import Generator
from distgen.writers import *
from pmd_beamphysics import ParticleGroup
from pmd_beamphysics.plot import marginal_plot

import ssnl 
from runOPAL import runOPAL

plt.rc('xtick',labelsize=12)
plt.rc('ytick',labelsize=12)
plt.rc('axes',labelsize=14)
plt.rc('axes', axisbelow=True)

# OPAL plotting
from opal.datasets.StatDataset import *

class StatWrapper:
    
    def __init__(self, directory, fname):
        self._stat = StatDataset(directory, fname)
    
    def __getitem__(self, variable):
        return self._stat.getData(variable)


def make_simulation_files(x, dvarkeys, base_filename):
    '''
    Make input simulation file for opal run.

    x             = input variable values
    dvarkeys      = inpt variable names
    base_filename = file name of opal simulation
    '''
    # SQUEEZE
    if type(x) is list:
        x = np.array(x)
    if type(x) is np.ndarray:
        pass
    #import pdb; pdb.set_trace()
    x = np.squeeze(x)
    arguments = []
    sim_dir   = [base_filename]
    # Composing variable names and x values to set up simulation
    for i,key in enumerate(dvarkeys):
       #print(i, key, x[i])
       #sys.stdout.flush()
       variable = key+'='+str(x[i])  #str('%.2f'%(x[i]))
       arguments.append(variable)
       sim_dir.append('_'+variable)
    #Generating simulation files
    arguments.append('CORES=2 --test' ) #--quiet')
    test = runOPAL.main(arguments)

    return None

#import pdb; pdb.set_trace()
xscale   = np.array([('RADIUS',1e-2), ('FWHM', 1.0), ('PHGUNB', 1.0), ('PHBUN', 1.0), ('GBUN',0.1), ('SF1', 1e-3), ('SF2', 1e-3), \
                    ('PHCM1', 1.0), ('GCM1', 1.0), \
                    ('PHCM2', 1.0), ('GCM2', 1.0), \
                    ('PHCM3', 1.0), ('GCM3', 1.0), \
                    ('PHCM4', 1.0), ('GCM4', 1.0), \
                ],dtype=[('name', 'U10'), ('scale', 'f4')])

x = np.array(  [ 46.56057936,  11.06428989,   6.51297153, -65.13686448,  23.75656481,
   56.14544348,  30.18347773,  -6.16649421,   7.81374636, -22.81613547,
   15.98451053, -27.79411711,   4.89824237, -39.78207529,  14.58535995])


xrad  = x[0]
xfwhm = x[1]
xopal = x[2:]*xscale['scale'][2:]

temp_dir = '/Users/nneveu/github/emittance_minimization/code/slac/paper_test/'
sim_dir  = '/Users/nneveu/github/emittance_minimization/code/slac/paper_test/'

names = xscale['name'][2:]

make_simulation_files(xopal,names, 'sc_inj_C1') 

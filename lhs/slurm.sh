#!/bin/bash 

#SBATCH --partition=shared
#
#SBATCH --job-name=test
#SBATCH --output=output-%j.txt
#SBATCH --error=output-%j.txt
#
#SBATCH --nodes 2 
#SBATCH --time=00:30:00

module remove openmpi
module load devtoolset/9
module list

# Export SLURM_EXACT because of the new behavior in 21.08
export SLURM_EXACT=1
#export SLURM_MEM_PER_NODE=0
export NUM_WORKERS=4

source ~/.bashrc
export PYTHONPATH=/gpfs/slac/staas/fs1/g/accelerator_modeling/nneveu/injectoropt/:$PYTHONPATH
export PYTHONPATH=/gpfs/slac/staas/fs1/g/accelerator_modeling/nneveu/software/libensemble/:$PYTHONPATH
export PYTHONPATH=/gpfs/slac/staas/fs1/g/accelerator_modeling/nneveu/software/distgen/:$PYTHONPATH
export PATH=/gpfs/slac/staas/fs1/g/accelerator_modeling/nneveu/software/OPAL/opal_mpich/bin/:$PATH

conda activate
conda activate /gpfs/slac/staas/fs1/g/accelerator_modeling/nneveu/software/libe

which mpiexec
#python latin_sample_sc.py --comms local --nworkers 9
#/gpfs/slac/staas/fs1/g/accelerator_modeling/nneveu/software/OPAL/opal_mpich/bin/mpiexec -n 9 python latin_sample_sc.py
#python call_vtmop_lhs_restart.py --comms local --nworkers 9
mpiexec -np $(($NUM_WORKERS+1)) -ppn $(($NUM_WORKERS+1)) python latin_sample_sc.py


#
# Print the date again -- when finished
echo Finished at: `date`

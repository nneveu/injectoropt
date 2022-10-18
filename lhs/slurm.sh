#!/bin/bash 

#SBATCH --partition=shared
#
#SBATCH --job-name=test
#SBATCH --output=output-%j.txt
#SBATCH --error=output-%j.txt
#
#SBATCH -N 1 
####SBATCH --exclusive

######SBATCH --ntasks=9
######SBATCH --cpus-per-task=5
#####SBATCH --mem-per-cpu=1g
#
#SBATCH --time=00:40:00

module remove openmpi
module load devtoolset/9
module list

source ~/.bashrc
export PYTHONPATH=/gpfs/slac/staas/fs1/g/accelerator_modeling/nneveu/emittance_minimization/code/:$PYTHONPATH
export PYTHONPATH=/gpfs/slac/staas/fs1/g/accelerator_modeling/nneveu/software/libensemble/:$PYTHONPATH
export PYTHONPATH=/gpfs/slac/staas/fs1/g/accelerator_modeling/nneveu/software/distgen/:$PYTHONPATH

conda activate
conda activate /gpfs/slac/staas/fs1/g/accelerator_modeling/nneveu/software/libe

export PATH=/gpfs/slac/staas/fs1/g/accelerator_modeling/nneveu/software/OPAL/opal_mpich/bin/:$PATH

# Export SLURM_EXACT because of the new behavior in 21.08
export SLURM_EXACT=1
#export SRUN_CPUS_PER_TASK=5
#export SLURM_MEM_PER_NODE=0

which mpirun
#/gpfs/slac/staas/fs1/g/accelerator_modeling/nneveu/software/OPAL/opal_mpich/bin/mpiexec -n 5 python call_aposmm.py
python latin_sample_sc.py --comms local --nworkers 7 
#srun --ntasks=$(($NUM_WORKERS+1)) --nodes=1 python $EXE

#
# Print the date again -- when finished
echo Finished at: `date`

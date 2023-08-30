#!/bin/sh
#SBATCH -J 1e5-1
#SBATCH --gpus=1
#SBATCH -N 1 -n 8
##SBATCH --ntasks-per-node=1 --cpus-per-task=8
#SBATCH -o %x.o%j
#SBATCH -e %x.e%j
echo Running on hosts
echo Time is `date`
echo Directory is $PWD
echo This job runs on the following nodes:
echo $SLURM_JOB_NODELIST
# Your conda environment
export OMP_NUM_THREADS=8

#module add cuda/11.0
#module add cudnn/8.1.1.33_CUDA11.0
module add gcc/9.3

#ATTENTION! HERE MUSTT BE ONE LINE,OR ERROR!
source ~/.bashrc
conda activate flax-gpu
cd $PWD
python3 ../run/optimizer.py

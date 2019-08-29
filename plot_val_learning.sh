#!/bin/bash
#SBATCH --no-requeue
#SBATCH --qos=normal
#SBATCH --partition=evanreed
#SBATCH --output=job_out_plot.log
#SBATCH --error=job_err_plot.log
#SBATCH --mem=64G
#SBATCH --job-name=plot_Ih
#SBATCH --ntasks-per-node=20
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1

# Modules for numpy, scipy, and matplotlib.
ml load py-numpy/1.14.3_py36
ml load py-scipy/1.1.0_py36
ml load py-matplotlib/2.2.2_py36
ml load py-pandas/0.23.0_py36
ml load py-scikit-learn/0.19.1_py36

srun python3 $HOME/steinhardt_ml/tuning.py

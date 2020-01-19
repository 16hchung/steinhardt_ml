#!/bin/bash
#SBATCH --no-requeue
#SBATCH --qos=normal
#SBATCH --partition=evanreed
#SBATCH --output=job_tsne_out.log
#SBATCH --error=job_tsne_err.log
#SBATCH --mem=64G
#SBATCH --job-name=tsne_many
#SBATCH --ntasks-per-node=20
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1

ml load system
ml load qt
ml load py-numpy/1.14.3_py36
ml load py-matplotlib/2.2.2_py36
ml load py-pandas/0.23.0_py36
ml load py-scikit-learn/0.19.1_py36

srun python3 -m 03_visualization.b_tSNE.compute -m 

# TO RUN: cd into steinhardt_ml and run $ sbatch 03_visualization/b_tSNE/compute_tsne_job.sh

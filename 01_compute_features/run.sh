#!/bin/bash
sbatch <<EOT
#!/bin/bash
#SBATCH --no-requeue
#SBATCH --qos=normal
#SBATCH --partition=evanreed
#SBATCH --output="synth_job_out"$1".log"
#SBATCH --error="synth_job_err"$1".log"
#SBATCH --mem=64G
#SBATCH --job-name="stein_Ih"$1
#SBATCH --ntasks-per-node=20
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1

# Modules for Ovito.
ml --force purge
ml load math
ml load devel
ml load py-numpy/1.18.1_py36
ml load viz
ml load py-pandas/0.23.0_py36
ml load system
ml load qt

export MPLBACKEND="agg"
export PYTHONPATH=
srun ../ovito-3.0.0-dev608-x86_64/bin/ovitos -m 01_compute_features.01_compute --latt $1 --pseudo_param .35
EOT

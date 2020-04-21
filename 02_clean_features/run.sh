#!/bin/bash
sbatch <<EOT
#!/bin/bash
#SBATCH --no-requeue
#SBATCH --qos=normal
#SBATCH --partition=evanreed
#SBATCH --output="logs/clean_out.log"
#SBATCH --error="logs/clean_err.log"
#SBATCH --mem=128G
#SBATCH --job-name=stein_Ih
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
srun ../ovito-3.0.0-dev608-x86_64/bin/ovitos -m 02_clean_features.compute --cat
EOT

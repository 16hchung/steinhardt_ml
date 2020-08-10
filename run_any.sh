#!/bin/bash
sbatch <<EOT
#!/bin/bash
#SBATCH --no-requeue
#SBATCH --qos=normal
#SBATCH --partition=evanreed
#SBATCH --output="logs/"$1"_out.log"
#SBATCH --error="logs/"$1"_err.log"
#SBATCH --mem=160G
#SBATCH --job-name="_"$1"_"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$2
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00

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
srun ../ovito-3.0.0-dev608-x86_64/bin/ovitos -m $3
EOT

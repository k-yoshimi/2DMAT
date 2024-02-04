#!/bin/sh

#SBATCH -p i8cpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 128
#SBATCH -t 00:10:00

set -e
/home/issp/materiapps/bin/issp-ucount py2dmat
module list

srun python3 /home/k0136/k013603/git/2DMAT/src/py2dmat_main.py input.toml

#!/bin/sh

#SBATCH -p i8cpu
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -c 64
#SBATCH -t 00:30:00

set -e
source /home/issp/materiapps/oneapi_compiler_classic-2023.0.0--openmpi-4.1.5/py2dmat/py2dmatvars.sh
/home/issp/materiapps/bin/issp-ucount py2dmat

srun python3 /home/k0136/k013603/git/2DMAT/src/py2dmat_main.py input.toml

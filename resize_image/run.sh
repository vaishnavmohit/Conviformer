#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mohit_vaishnav@brown.edu
#SBATCH -n 5
#SBATCH --account=carney-tserre-condo
#SBATCH --mem=5G
#SBATCH -N 1
# Specify an output file
#SBATCH -o %j.out
#SBATCH -e %j.err

# Specify a job name:
#SBATCH --time=30:00:00

# activate conda env
module load cuda/11.1.1
module load gcc/8.3
source ~/data/data/mvaishn1/env/fossil/bin/activate
module load python/3.7.4

python padherb2021.py $1 $2
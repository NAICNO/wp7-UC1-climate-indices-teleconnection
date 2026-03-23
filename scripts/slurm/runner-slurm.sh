#!/bin/sh

#SBATCH --nodes=1 
#SBATCH --mail-type=BEGIN
#SBATCH --priority=low 

source $7
cd $3
export PRED_LEN=$1
export DATA_PATHS=$2
export FOLDERPATH=$3
export PREDICTOR_NAMES=$4
export ISFULLLENGTH=$6
export WINDOWINSIZE=300 # default 150
export WINDOWINGSTEP_NUMBER=30
export ENSEMBLES_NUMBER=20
ALGORITHM_TYPE=$8

if [ "$6" = "True" ]; then
    export OUTPOTLOGS="${ALGORITHM_TYPE}_search_fulllength_$5"
else
    export OUTPOTLOGS="${ALGORITHM_TYPE}_search_sorterterm_$5"
fi
-
python scripts/teleconnection/${ALGORITHM_TYPE}_search.py
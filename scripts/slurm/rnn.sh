#!/bin/bash

# Array of the second parameters
params=(
    "amoSSTmjjaso"
    "glTSjfm"
    "nhTSjfm"
    "shTSjfm"
    "cetTSjfm"
    "satlSSTjfm"
    "ensoSSTjfm"
    "neurTSjfm"
    "seurTSjfm"
    "arcTSjfm"
    "nhSICjfm"
    "nhSICsep"
    "shSICmar"
    "shSICsep"
    "nchinaPRjja"
    "yrvPRjja"
    "ismPRjja"
    "wnorPRjfm"
    "naoPSLjfm"
    "eapPSLjfm"
    "scpPSLjfm"
    "eurPSLjfm"
    "solFORC"
    "samPSLann"
    "atlSICjfm"
    "atlSICsep"
    "pdoSSTjfm"
    "nppc2SSTjfm"
    "nppc3SSTjfm"
    "AMOCann"
    "traBO"
    "traBS"
    "traCA"
    "traDS"
    "traDP"
    "traEC"
    "traEU"
    "traFSC"
    "traFB"
    "traFS"
    "traIFC"
    "traIT"
    "traMC"
    "traTLS"
    "traWP"
    "alpi"
    "mhfATL30"
    "mhfATL45"
    "mhfATL60"
    "mhfIP30"
    "tripol1"
    "tripol2"
    "tripol3"
    "amo1"
    "amo2"
    "amo3"
    "labSSTjfm"
    "ormenSSTjfm"
    "gyreSSTjfm"
    "ginwSSTjfm"
    "glomaSSTjfm"
    "stratZ50ndj01"
    "stratZ50ndj02"
    "stratZ50ndj03"
    "stratZ50ndj04"
)

params=('glomaSSTjfm' 'nhSICsep' 'nhTSjfm' 'shTSjfm' 'amoSSTmjjaso' 'glTSjfm' 'amo2' 'amo1' 'amo3')
# Array of the datasets
datasets=(
 #   "dataset/noresm-f-p1000_picntrl_new_jfm.csv"
    "dataset/noresm-f-p1000_slow_new_jfm.csv"
    "dataset/noresm-f-p1000_shigh_new_jfm.csv"
)

# Array of the sixth parameters
iterations=(10 20)

# Array of split values
splits=(
    0.6
    0.8
   # 0.4
   # 0.2
)

mls=(
    "LRDropoutPSO"
    "LRDropoutPSO_50percent"
    "LRforcedPSO_10percent"
    "LRforcedPSO_75percent"
    #"dataset/noresm-f-p1000_shigh_new_jfm.csv"
)

# Loop through all combinations of parameters, datasets, splits, and iterations
for mlalgo in "${mls[@]}"; do
    for split in "${splits[@]}"; do
        for dset in "${datasets[@]}"; do
            for param in "${params[@]}"; do
                for iter in "${iterations[@]}"; do
                    sbatch --partition=comp parameterized_job.slurm "$param" "$mlalgo" "$split" "$dset" "$iter"
                done
            done
        done
    done
done

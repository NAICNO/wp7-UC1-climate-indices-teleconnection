#!/bin/bash
#SBATCH --job-name=forecast_job
#SBATCH --output=slurm-logs/%j.log
#SBATCH --error=slurm-logs/%j.log
#SBATCH --time=4-10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2  # Adjust based on your needs
#SBATCH --mem=3G  # Adjust based on your needs

# Load necessary modules or activate virtual environment
source /cluster/home/hasv/activate.sh
cd /cluster/home/hasv/project-files/teleconnections/d7.2-Use-case1

# Run the Python script with parameters
python scripts/lrbased_teleconnection/main.py --target_feature=$1 --modelname=$2 --splitsize=$3 --step_lag=$5  --data_file=$4 --max_allowed_features=6 --with_mean_feature  
python scripts/lrbased_teleconnection/main.py --target_feature=$1 --modelname=$2 --splitsize=$3 --step_lag=$5  --data_file=$4 --max_allowed_features=6 --with_mean_feature --delete_features=glTSjfm,nhTSjfm,shTSjfm,cetTSjfm,satlSSTjfm,ensoSSTjfm,neurTSjfm,seurTSjfm,arcTSjfm,nchinaPRjja,yrvPRjja,ismPRjja,wnorPRjfm,samPSLann,pdoSSTjfm,nppc3SSTjfm,traBO,traBS,traCA,traDS,traDP,traEC,traEU,traFSC,traFB,traFS,traIFC,traIT,traMC,traTLS,traWP,alpi,mhfATL30,mhfATL45,mhfATL60,mhfIP30,tripol1,tripol2,tripol3,labSSTjfm,ormenSSTjfm,gyreSSTjfm,ginwSSTjfm,glomaSSTjfm  

#!/bin/bash -l
export DATE=$(date +%Y%m%d_%H%M%S)
export LABEL="$2"
export JOB_NAME=optimize_dentate_circuit_model_"$LABEL"_"$DATE"
export CONFIG_FILE_PATH="$1"
export SEED="$3"
sbatch <<EOT
#!/bin/bash -l
#SBATCH -J $JOB_NAME
#SBATCH -o /scratch1/06441/aaronmil/logs/optimize_dentate_circuit_model/$JOB_NAME.%j.o
#SBATCH -e /scratch1/06441/aaronmil/logs/optimize_dentate_circuit_model/$JOB_NAME.%j.e
#SBATCH -p normal
#SBATCH -N 4
#SBATCH -n 224
#SBATCH -t 6:00:00
#SBATCH --mail-user=neurosutras@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

set -x

cd $WORK2/dentate_circuit_model

ibrun -n 224 python3 -m nested.optimize --config-file-path=$CONFIG_FILE_PATH \
  --output-dir=$SCRATCH/data/optimize_dentate_circuit_model --pop_size=200 --max_iter=50 --path_length=3 --disp \
  --label=$LABEL --seed=$SEED
EOT

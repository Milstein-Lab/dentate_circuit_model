#!/bin/bash -l
export PARAM_FILE_PATH=$1
export EXPORT_FILE_PATH_BASE=$2

declare -a config_file_names=(optimize_config_0_input_output_multiple_seeds.yaml
  optimize_config_1_input_output_multiple_seeds.yaml
  optimize_config_2_FF_Inh_multiple_seeds.yaml
  optimize_config_3_FB_Inh_multiple_seeds.yaml
  optimize_config_4_FF+FB_Inh_multiple_seeds.yaml
  optimize_config_5_FF_Inh+indirect_FB_Inh_multiple_seeds.yaml
  optimize_config_5c_FF_Inh+indirect_FB_Inh_multiple_seeds.yaml
  optimize_config_6_FF_Inh+indirect_FB_Inh+FB_Exc_multiple_seeds.yaml
  optimize_config_6b_FF_Inh+indirect_FB_Inh+FB_Exc_multiple_seeds.yaml)

declare -a model_keys=(0 1 2 3 4 5 5c 6 6b)

arraylength=${#config_file_names[@]}

for ((i=0; i<${arraylength}; i++))
do
  mpirun -n 6 python -m mpi4py.futures -m nested.analyze --config-file-path=config/${config_file_names[$i]} \
    --framework=mpi --export --export-file-path=${EXPORT_FILE_PATH_BASE}.hdf5 --param-file-path=$PARAM_FILE_PATH \
    --model-key=${model_keys[$i]}

done

mpirun -n 6 python -m mpi4py.futures -m nested.analyze \
  --config-file-path=config/optimize_config_6b_FF_Inh+indirect_FB_Inh+FB_Exc_multiple_seeds.yaml \
  --param-file-path=$PARAM_FILE_PATH --model-key=6b --framework=mpi --export --export_dynamics \
  --export-file-path=${EXPORT_FILE_PATH_BASE}_dynamics.hdf5


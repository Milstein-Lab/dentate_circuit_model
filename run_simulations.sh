#!/bin/bash -l
export CONFIG_DIR=$SCRATCH/dentate_circuit_model/config
export DATA_DIR=$SCRATCH/dentate_circuit_model/data

declare -a config_paths=($CONFIG_DIR/optimize_config_0_input_output_multiple_seeds.yaml)

arraylength=${#config_paths[@]}

for ((i=0; i<${arraylength}; i++))
do
  sh mpirun -n 6 python -m mpi4py.futures -m nested.analyze --config-file-path=${config_paths[$i]} --framework=mpi --export --export-file-path=$DATA_DIR/dentate_optimization.hdf5
done

#mpirun -n 6 python -m mpi4py.futures -m nested.analyze --config-file-path=config/optimize_config_0_input_output_multiple_seeds.yaml --param-file-path=config/20211110_model_params.yaml --model-key=0 --framework=mpi --export --export-file-path=data/dentate_optimization.hdf5
#mpirun -n 6 python -m mpi4py.futures -m nested.analyze --config-file-path=config/optimize_config_1_input_output_multiple_seeds.yaml --param-file-path=config/20211110_model_params.yaml --model-key=1 --framework=mpi --export --export-file-path=data/dentate_optimization.hdf5
#mpirun -n 6 python -m mpi4py.futures -m nested.analyze --config-file-path=config/optimize_config_2_FF_Inh_multiple_seeds.yaml --param-file-path=config/20211110_model_params.yaml --model-key=2 --framework=mpi --export --export-file-path=data/dentate_optimization.hdf5
#mpirun -n 6 python -m mpi4py.futures -m nested.analyze --config-file-path=config/optimize_config_3_FB_Inh_multiple_seeds.yaml --param-file-path=config/20211110_model_params.yaml --model-key=3 --framework=mpi --export --export-file-path=data/dentate_optimization.hdf5
#mpirun -n 6 python -m mpi4py.futures -m nested.analyze --config-file-path=config/optimize_config_4_FF+FB_Inh_multiple_seeds.yaml --param-file-path=config/20211110_model_params.yaml --model-key=4 --framework=mpi --export --export-file-path=data/dentate_optimization.hdf5
#mpirun -n 6 python -m mpi4py.futures -m nested.analyze --config-file-path=config/optimize_config_5_FF_Inh+indirect_FB_Inh_multiple_seeds.yaml --param-file-path=config/20211110_model_params.yaml --model-key=5 --framework=mpi --export --export-file-path=data/dentate_optimization.hdf5
#mpirun -n 6 python -m mpi4py.futures -m nested.analyze --config-file-path=config/optimize_config_5c_FF_Inh+indirect_FB_Inh_multiple_seeds.yaml --param-file-path=config/20211110_model_params.yaml --model-key=5c --framework=mpi --export --export-file-path=data/dentate_optimization.hdf5
#mpirun -n 6 python -m mpi4py.futures -m nested.analyze --config-file-path=config/optimize_config_6_FF_Inh+indirect_FB_Inh+FB_Exc_multiple_seeds.yaml --param-file-path=config/20211110_model_params.yaml --model-key=6 --framework=mpi --export --export-file-path=data/dentate_optimization.hdf5
#mpirun -n 6 python -m mpi4py.futures -m nested.analyze --config-file-path=config/optimize_config_6b_FF_Inh+indirect_FB_Inh+FB_Exc_multiple_seeds.yaml --param-file-path=config/20211110_model_params.yaml --model-key=6b --framework=mpi --export --export-file-path=data/dentate_optimization.hdf5
#!/bin/bash -l

sh optimize_dentate_circuit_model_multiple_seeds_frontera.sh \
  config/optimize_config_2_FF_Inh_multiple_seeds_no_sel_loss.yaml FF_Inh 10

declare -a weight_seeds=(1234 1235 1236 1237 1238)

declare -a opt_seeds=(11 12 13 14 15)

arraylength=${#weight_seeds[@]}

for ((i=0; i<${arraylength}; i++))
do
  sh optimize_dentate_circuit_model_frontera_one_seed.sh \
    config/optimize_config_2_FF_Inh.yaml FF_Inh_${weight_seeds[$i]} ${opt_seeds[$i]} ${weight_seeds[$i]}
done

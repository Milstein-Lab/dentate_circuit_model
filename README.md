Neural circuit regulation of sparsity and pattern separation in the dentate gyrus
========

In this project, we construct a series of simple models of the dentate gyrus to explore how different cell types and the
distributions of connectivity between them influence population coding and information processing in the network. These
are quantified in terms of sparsity, selectivity, and discriminability at the population level within each cell type. 


How to run a single simulation
------------

### Requirements
Install requirements by running:

    $ pip install -r requirements.txt

### Usage
Download this repository by cloning it from github:

    $ git clone https://github.com/Milstein-Lab/dentate_circuit_model

A single instance of a model simulation can then be run from the terminal/command line by navigating to the directory
where the repository has been saved an executing one of the following commands:

Models in Figure 1 & 2:
* Input-Output with uniform weights:


        $ python simulate_dynamic_model.py --config_file_path=config/simulate_config_0_input_output.yaml --plot

* Input-Output with log-normal weights:


        $ python simulate_dynamic_model.py --config_file_path=config/simulate_config_1_input_output.yaml --plot

Models in Figure 3:
* FF Inhibition:

        $ python simulate_dynamic_model.py --config_file_path=config/simulate_config_2_FF_Inh.yaml --plot

* FF Inhibition, optimized without selectivity constraint:

        $ python simulate_dynamic_model.py --config_file_path=config/simulate_config_2_FF_Inh_no_sel_loss.yaml --plot`

Models in Figure 4:
* FB Inhibition:

        $ python simulate_dynamic_model.py --config_file_path=config/simulate_config_3_FB_Inh.yaml --plot

* FF + FB Inhibition:

        $ python simulate_dynamic_model.py --config_file_path=config/simulate_config_4_FF+FB_Inh.yaml --plot

Models in Figure 5:
* FF Inhibition + indirect FB Inhibition (via excitatory interneurons):\
`python simulate_dynamic_model.py --config_file_path=config/simulate_config_5_FF_Inh+indirect_FB_Inh.yaml --plot`
* (-) recurrent connections among the excitatory interneurons:\
`python simulate_dynamic_model.py --config_file_path=config/simulate_config_5_FF_Inh+indirect_FB_Inh_no_recurrence.yaml --plot`
* (+) direct FB exitation from Mossy cells:\
`python simulate_dynamic_model.py --config_file_path=config/simulate_config_6_FF_Inh+indirect_FB_Inh+FB_Exc.yaml --plot`

#### Optional flags to add at the end of the command:
* `--plot`
* `--export`: generates hdf5 files containing simulation data
* `--export_file_name`: for saving simulation data to a specific file (by default will use date and time as filename)
* `--data_dir:` for saving simulation data to a specific directory (by default will save to /data)


How to run and optimize multiple simulations with parallelization
------------
### Requirements
In addition to the requirements above, running simulations with parallelization requires installing:

* [MPI for Python](https://mpi4py.readthedocs.io/en/stable/install.html)
* The [nested](https://github.com/neurosutras/nested) parallel computing package:


      $ git clone https://github.com/neurosutras/nested.git

After cloning [nested](https://github.com/neurosutras/nested), add its directory to the PYTHONPATH on your machine


### Usage
To run and export data for all simulations used in the paper (with multiple random seeds for each model), execute the following shell script:

    $ sh export_model_data.sh config/20211116_model_params.yaml data/exported_dentate_model_data

To optimize parameters for a single model configuration, execute one of the following commands:
(**Note** this requires significant processing power. Smaller optimizations can be run by reducing the pop_size, max_iter, and path_length)

Models in Figure 1 & 2:
* Input-Output with uniform weights:\
`mpirun -n 6 python -m mpi4py.futures -m nested.optimize --config-file-path=config/optimize_config_0_input_output_multiple_seeds.yaml --path_length=3 --max_iter=50 --pop_size=200 --disp --framework=mpi`
* Input-Output with log-normal weights:\
`mpirun -n 6 python -m mpi4py.futures -m nested.optimize --config-file-path=config/optimize_config_1_input_output_multiple_seeds.yaml --path_length=3 --max_iter=50 --pop_size=200 --disp --framework=mpi`

Models in Figure 3:
* FF Inhibition: \
`mpirun -n 6 python -m mpi4py.futures -m nested.optimize --config-file-path=config/optimize_config_2_FF_Inh_multiple_seeds.yaml --path_length=3 --max_iter=50 --pop_size=200 --disp --framework=mpi`
* FF Inhibition, optimized without selectivity constraint:\ 
`mpirun -n 6 python -m mpi4py.futures -m nested.optimize --config-file-path=config/optimize_config_2_FF_Inh_multiple_seeds_no_sel_loss.yaml --path_length=3 --max_iter=50 --pop_size=200 --disp --framework=mpi`

Models in Figure 4:
* FB Inhibition:\
`mpirun -n 6 python -m mpi4py.futures -m nested.optimize --config-file-path=config/optimize_config_3_FB_Inh_multiple_seeds.yaml --path_length=3 --max_iter=50 --pop_size=200 --disp --framework=mpi`
* FF + FB Inhibition:\
`mpirun -n 6 python -m mpi4py.futures -m nested.optimize --config-file-path=config/optimize_config_4_FF+FB_Inh_multiple_seeds.yaml --path_length=3 --max_iter=50 --pop_size=200 --disp --framework=mpi`

Models in Figure 5:
* FF Inhibition + indirect FB Inhibition (via excitatory interneurons):\
`mpirun -n 6 python -m mpi4py.futures -m nested.optimize --config-file-path=config/optimize_config_5_FF_Inh+indirect_FB_Inh_multiple_seeds.yaml --path_length=3 --max_iter=50 --pop_size=200 --disp --framework=mpi`
* (-) recurrent connections among the excitatory interneurons:\
`mpirun -n 6 python -m mpi4py.futures -m nested.optimize --config-file-path=config/optimize_config_5c_FF_Inh+indirect_FB_Inh_multiple_seeds.yaml --path_length=3 --max_iter=50 --pop_size=200 --disp --framework=mpi`
* (+) direct FB exitation from Mossy cells:\
`mpirun -n 6 python -m mpi4py.futures -m nested.optimize --config-file-path=config/optimize_config_6_FF_Inh+indirect_FB_Inh+FB_Exc_multiple_seeds.yaml --path_length=3 --max_iter=50 --pop_size=200 --disp --framework=mpi`


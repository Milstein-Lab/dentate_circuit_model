# This script is the same as "optimize_dynamic_model.py" but with references to nested and
# related parallel computing functions removed to minimize the requirements for running a single model

from utils import *


#############################################################################
"""
Main function:
- used to run a single instance of a model from the command line.
- requires a yaml file containing the relevant model parameters.

Example command to run from terminal:
python optimize_dynamic_model.py --config_file_path=config/simulate_config_2_FF_Inh.yaml --plot 
"""

@click.command()
@click.option("--config_file_path", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--export_file_name", type=str, default=None)
@click.option("--data_dir", type=click.Path(exists=True, file_okay=False, dir_okay=True), default='data')
@click.option("--plot", is_flag=True)
@click.option("--export", is_flag=True)
@click.option("--fast", type=bool, default=False)
def main(config_file_path, export_file_name, data_dir, plot, export, fast):
    """
    Given model configuration parameters, build a network, run a simulation and analyze the output.
    Optionally can generate summary plots and/or export data to an hdf5 file.
    :param config_file_path: str name of .yaml file containing configuration parameters
    :param export_file_name: str; hdf5 file name for data export
    :param data_dir: str (path); directory to export data
    :param plot: bool; whether to generate plots
    :param export: bool; whether to export data to hdf5
    :param fast: bool; whether to sacrifice accuracy for speed
    """
    start_time = time.time()
    parameter_dict = read_from_yaml(config_file_path)
    description = parameter_dict['description']

    num_units_dict = parameter_dict['num_units_dict']
    num_input_units = num_units_dict['Input']
    train_epochs = parameter_dict['train_epochs']
    train_seed = parameter_dict['train_seed']

    # generate all possible binary input patterns with specified number units in the input layer
    sorted_input_patterns = get_binary_input_patterns(num_input_units, sort=True, plot=False)

    num_units_dict['Output'] = len(sorted_input_patterns)

    duration = parameter_dict['duration']
    activation_function_dict = parameter_dict['activation_function_dict']
    weight_config_dict = parameter_dict['weight_config_dict']
    weight_seed = parameter_dict['weight_seed']
    cell_tau_dict = parameter_dict['cell_tau_dict']
    synapse_tau_dict = parameter_dict['synapse_tau_dict']
    dt = parameter_dict['dt']
    time_point = parameter_dict['time_point']
    synaptic_reversal_dict = parameter_dict['synaptic_reversal_dict']

    # Generate weight dictionary specifying all conncections based on the specified parameters
    weight_dict = get_weight_dict(num_units_dict, weight_config_dict, weight_seed, description=description, plot=plot)

    # Simulate network dynamics
    t = np.arange(0., duration + dt / 2., dt) # initial test
    if train_epochs > 0:
        # train step
        current_time = time.time()
        channel_conductance_dynamics_dict, net_current_dynamics_dict, cell_voltage_dynamics_dict, \
        network_activity_dynamics_dict, train_network_activity_history_dict, weight_history_dict = \
            train_network(t, sorted_input_patterns, num_units_dict, synapse_tau_dict, cell_tau_dict, weight_dict,
                          weight_config_dict, activation_function_dict, synaptic_reversal_dict, time_point,
                          train_epochs, train_seed, disp=True, fast=fast)
        print('Train took %.1f s' % (time.time() - current_time))
    else:
        weight_history_dict = None

    # final test
    current_time = time.time()
    channel_conductance_dynamics_dict, net_current_dynamics_dict, cell_voltage_dynamics_dict, \
    network_activity_dynamics_dict, mean_network_activity_dict = \
        test_network(t, sorted_input_patterns, num_units_dict, synapse_tau_dict, cell_tau_dict, weight_dict,
                     weight_config_dict, activation_function_dict, synaptic_reversal_dict, time_point, fast=fast)
    final_activity_dict = deepcopy(mean_network_activity_dict)
    print('Test took %.1f s' % (time.time() - current_time))

    sparsity_dict, similarity_matrix_dict, selectivity_dict, fraction_active_patterns_dict, \
    fraction_active_units_dict = analyze_slice(mean_network_activity_dict)

    sparsity_dynamics_dict, similarity_dynamics_dict, selectivity_dynamics_dict, \
    fraction_nonzero_response_dynamics_dict = analyze_dynamics(network_activity_dynamics_dict)

    median_sparsity_dynamics_dict, median_similarity_dynamics_dict, mean_selectivity_dynamics_dict, \
    fraction_nonzero_response_dynamics_dict = analyze_median_dynamics(network_activity_dynamics_dict)

    if export:
        if export_file_name is None:
            export_file_name = '%s_exported_model_data.hdf5' % datetime.datetime.today().strftime('%Y%m%d_%H%M%S')
        export_file_path = '%s/%s' % (data_dir, export_file_name)

        model_config_dict = {'duration': duration,
                             'dt': dt
                             }

        export_model_slice_data(export_file_path, description, weight_seed, model_config_dict,
                                weight_dict, num_units_dict, activation_function_dict,
                                weight_config_dict, mean_network_activity_dict, weight_history_dict)

        export_file_path = export_file_path[:-5] + '_dynamics.hdf5'
        export_dynamic_model_data(export_file_path, description, weight_seed, model_config_dict, num_units_dict,
                                  activation_function_dict, weight_config_dict, weight_dict, cell_tau_dict,
                                  synapse_tau_dict, channel_conductance_dynamics_dict, net_current_dynamics_dict,
                                  cell_voltage_dynamics_dict, network_activity_dynamics_dict)

    print('Simulation took %.1f s' % (time.time() - start_time))

    if plot:
        plot_dynamics(t, median_sparsity_dynamics_dict, median_similarity_dynamics_dict, mean_selectivity_dynamics_dict,
                      fraction_nonzero_response_dynamics_dict, description)

        plot_model_summary(mean_network_activity_dict, sparsity_dict, similarity_matrix_dict, selectivity_dict,
                           description)
        # this forces all plots generated with fig.show() to wait for the user to close them before exiting python
        plt.show()

    globals().update(locals())


if __name__ == '__main__':
    # this extra flag stops the click module from forcing system exit when python is in interactive mode
    main(standalone_mode=False)
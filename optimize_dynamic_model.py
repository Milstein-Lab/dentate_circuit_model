from utils import *
from nested.utils import read_from_yaml, Context, param_array_to_dict, str_to_bool
from simulate_dynamic_model import train_network, Hebb_weight_norm, plain_Hebb


context = Context()


#############################################################################
# Configure model for nested optimization

# Example command to run from terminal:
# python -m nested.analyze --config_file_path=$PATH_TO_CONFIG_YAML --disp --framework=serial --plot \
#    --param-file-path=$PATH_TO_PARAM_YAML --model-key=$KEY_TO_PARAM_YAML

# python -m nested.optimize --config-file-path=config/optimize_config_2_FF_Inh.yaml --path_length=1 --max_iter=1 \
#   --pop_size=1 --disp --framework=serial --interactive

def config_worker():
    num_input_units = context.num_units_dict['Input']

    # generate all possible binary input patterns with specified number units in the input layer
    sorted_input_patterns = get_binary_input_patterns(num_input_units, sort=True, plot=context.plot_patterns)

    context.duration = float(context.duration)

    t = np.arange(0., context.duration + context.dt / 2., context.dt)
    
    context.train_epochs = int(context.train_epochs)
    context.train_seed = int(context.train_seed)
    context.num_instances = int(context.num_instances)
    
    if 'plot' not in context():
        context.plot = False

    if 'debug' not in context():
        context.debug = False
    
    if 'verbose' not in context():
        context.verbose = False
    else:
        context.verbose = str_to_bool(context.verbose)
    
    if 'export_dynamics' not in context():
        context.export_dynamics = False
    elif isinstance(context.export_dynamics, str):
        context.export_dynamics = bool(strtobool(context.export_dynamics))

    if 'export_dynamics_light' not in context():
        context.export_dynamics_light = False
    elif isinstance(context.export_dynamics_light, str):
        context.export_dynamics_light = bool(strtobool(context.export_dynamics_light))

    if 'allow_fail' not in context():
        context.allow_fail = True
    elif isinstance(context.allow_fail, str):
        context.allow_fail = bool(strtobool(context.allow_fail))

    context.update(locals())


def modify_network(param_dict):
    for param_name, param_val in param_dict.items():
        parsed_param_name = param_name.split(';')
        if parsed_param_name[0] == 'mean_weight':
            post_pop_name = parsed_param_name[1]
            pre_pop_name = parsed_param_name[2]
            context.weight_config_dict[post_pop_name][pre_pop_name]['mean_magnitude'] = param_val
        elif parsed_param_name[0] == 'learning_rate':
            if len(parsed_param_name) > 1:
                post_pop_name = parsed_param_name[1]
                pre_pop_name = parsed_param_name[2]
                context.weight_config_dict[post_pop_name][pre_pop_name]['learning_rule_params'][
                    'learning_rate'] = param_val
            else:
                for post_pop_name in context.weight_config_dict:
                    for pre_pop_name in context.weight_config_dict[post_pop_name]:
                        if ('learning_rule_parms' in
                                context.weight_config_dict[post_pop_name][pre_pop_name] and
                                'learning_rate' in
                                context.weight_config_dict[post_pop_name][pre_pop_name]['learning_rule_params']):
                            context.weight_config_dict[post_pop_name][pre_pop_name]['learning_rule_params'][
                                'learning_rate'] = param_val


def compute_features(param_array, model_id=None, export=False, plot=False):
    """

    :param x: array of float
    :param model_id: str
    :param export: bool
    :return: dict
    """
    return compute_features_multiple_instances(param_array, context.weight_seed, model_id, export)


def get_objectives(orig_features_dict, model_id=None, export=False, plot=False):
    """
    Compute loss function.
    :param org_features_dict: dict
    :param model_id: str
    :param export: bool
    :param plot: bool
    :return: tuple of dict
    """

    sparsity_errors = (context.target_val['sparsity'] -
                       orig_features_dict['sparsity_array'])/context.target_range['sparsity']

    # penalize discriminability of silent patterns
    bad_indexes = np.where(np.isnan(orig_features_dict['similarity_array']))
    orig_features_dict['similarity_array'][bad_indexes] = 1.
    discriminability_errors = (context.target_val['similarity'] -
                               orig_features_dict['similarity_array'])/context.target_range['similarity']

    selectivity_errors = (context.target_val['selectivity'] -
                          orig_features_dict['selectivity_array'])/context.target_range['selectivity']
    fraction_active_patterns_error = (context.target_val['fraction_active_patterns'] -
                              orig_features_dict['fraction_active_patterns']) / \
                            context.target_range['fraction_active_patterns']
    fraction_active_units_error = (context.target_val['fraction_active_units'] -
                             orig_features_dict['fraction_active_units']) / \
                            context.target_range['fraction_active_units']

    objectives_dict = {'sparsity_loss': np.sum(sparsity_errors**2),
                       'discriminability_loss': np.nansum(discriminability_errors**2),
                       'selectivity_loss': np.sum(selectivity_errors**2),
                       'fraction_active_patterns_loss': fraction_active_patterns_error**2,
                       'fraction_active_units_loss': fraction_active_units_error**2}

    summary_features_dict = {'sparsity': np.mean(orig_features_dict['sparsity_array']),
                             'similarity': np.nanmean(orig_features_dict['similarity_array']),
                             'selectivity': np.mean(orig_features_dict['selectivity_array']),
                             'fraction_active_patterns': orig_features_dict['fraction_active_patterns'],
                             'fraction_active_units': orig_features_dict['fraction_active_units']}

    return summary_features_dict, objectives_dict


# When simulating with multiple random seeds in parallel, use alternative compute functions:
def get_weight_seeds():
    weight_seed_list = list(range(int(context.init_weight_seed),
                                  int(context.init_weight_seed) + int(context.num_instances)))
    return [weight_seed_list]


def compute_features_multiple_instances(param_array, weight_seed, model_id=None, export=False, plot=False):
    """

    :param param_array: array of float
    :param weight_seed: int
    :param model_id: str
    :param export: bool
    :return: dict
    """
    start_time = time.time()

    param_dict = param_array_to_dict(param_array, context.param_names)
    modify_network(param_dict) #update the weight config dict

    weight_seed = int(weight_seed)

    weight_dict = get_weight_dict(context.num_units_dict, context.weight_config_dict, weight_seed,
                                  description=context.description, plot=plot)
    
    if context.train_epochs > 0:
        # train
        current_time = time.time()
        channel_conductance_dynamics_dict, net_current_dynamics_dict, cell_voltage_dynamics_dict, \
            network_activity_dynamics_dict, train_network_activity_history_dict, weight_history_dict = \
            train_network(context.t, context.sorted_input_patterns, context.num_units_dict, context.synapse_tau_dict,
                          context.cell_tau_dict, weight_dict, context.weight_config_dict,
                          context.activation_function_dict, context.synaptic_reversal_dict, context.time_point,
                          context.train_epochs, context.train_seed, context.verbose)
        if context.verbose:
            print('Model_id: %s; Training took %.1f s' % (model_id, time.time() - current_time))
    else:
        weight_history_dict = None
    
    # test after train
    current_time = time.time()
    channel_conductance_dynamics_dict, net_current_dynamics_dict, cell_voltage_dynamics_dict, \
    network_activity_dynamics_dict = \
        get_network_dynamics_dicts(context.t, context.sorted_input_patterns, context.num_units_dict,
                                   context.synapse_tau_dict, context.cell_tau_dict,
                                   weight_dict, context.weight_config_dict, context.activation_function_dict,
                                   context.synaptic_reversal_dict)
    if context.verbose:
        print('Model id: %s; Test took %.1f s' % (model_id, time.time() - current_time))
    
    network_activity_dict = slice_network_activity_dynamics_dict(network_activity_dynamics_dict, context.t,
                                                                 time_point=context.time_point)

    sparsity_dict, similarity_matrix_dict, selectivity_dict, fraction_active_patterns_dict, \
    fraction_active_units_dict = analyze_slice(network_activity_dict)

    # extract all values below diagonal
    similarity_matrix_idx = np.tril_indices_from(similarity_matrix_dict['Output'], -1)

    # Generate dictionary for "features" that will be used in the loss function (get objectives)
    orig_features_dict = {'sparsity_array': sparsity_dict['Output'],
                          'similarity_array': similarity_matrix_dict['Output'][similarity_matrix_idx],
                          'selectivity_array': selectivity_dict['Output'],
                          'fraction_active_patterns': fraction_active_patterns_dict['Output'],
                          'fraction_active_units': fraction_active_units_dict['Output']}

    if export:
        model_config_dict = {'duration': context.duration,
                             'dt': context.dt}

        if context.export_dynamics:
            export_dynamic_model_data(context.temp_output_path, context.description, weight_seed, model_config_dict,
                                      context.num_units_dict, context.activation_function_dict,
                                      context.weight_config_dict, weight_dict, context.cell_tau_dict,
                                      context.synapse_tau_dict, channel_conductance_dynamics_dict,
                                      net_current_dynamics_dict, cell_voltage_dynamics_dict,
                                      network_activity_dynamics_dict)

        elif context.export_dynamics_light:
            export_dynamic_activity_data(context.temp_output_path, context.description, weight_seed, model_config_dict,
                                         context.num_units_dict, context.activation_function_dict,
                                         network_activity_dynamics_dict)

        else:
            export_model_slice_data(context.temp_output_path, context.description, weight_seed, model_config_dict,
                                    weight_dict, context.num_units_dict, context.activation_function_dict,
                                    context.weight_config_dict, network_activity_dict, weight_history_dict)

    if plot:
        plot_model_summary(network_activity_dict, sparsity_dict, similarity_matrix_dict,
                           selectivity_dict, context.description)

        median_sparsity_dynamics_dict, median_similarity_dynamics_dict, mean_selectivity_dynamics_dict, \
        fraction_nonzero_response_dynamics_dict = analyze_median_dynamics(network_activity_dynamics_dict)

        plot_dynamics(context.t,
                      median_sparsity_dynamics_dict,
                      median_similarity_dynamics_dict,
                      mean_selectivity_dynamics_dict,
                      fraction_nonzero_response_dynamics_dict,
                      context.description)

    if context.debug:
        print('%s used weight_seed: %i' % (os.getpid(), weight_seed))
        print('Simulation took %.1f s' % (time.time() - start_time))
        sys.stdout.flush()
        context.update(locals())

    if context.allow_fail:
        for pop_name in fraction_active_units_dict:
            if pop_name in context.fraction_active_patterns_threshold:
                this_fraction_active_patterns_threshold = context.fraction_active_patterns_threshold[pop_name]
            else:
                this_fraction_active_patterns_threshold = context.fraction_active_patterns_threshold['default']
            if fraction_active_patterns_dict[pop_name] < this_fraction_active_patterns_threshold:
                print('pid: %i; model_id: %s failed; description: %s, weight_seed: %i; population: %s did not meet'
                      ' fraction_active_patterns_threshold: %.2f' %
                      (os.getpid(), model_id, context.description, weight_seed, pop_name,
                       this_fraction_active_patterns_threshold))
                sys.stdout.flush()
                return dict()

            if pop_name in context.fraction_active_units_threshold:
                this_fraction_active_units_threshold = context.fraction_active_units_threshold[pop_name]
            else:
                this_fraction_active_units_threshold = context.fraction_active_units_threshold['default']
            if fraction_active_units_dict[pop_name] < this_fraction_active_units_threshold:
                print('pid: %i; model_id: %s failed; description: %s, weight_seed: %i; population: %s did not meet'
                      ' fraction_active_units_threshold: %.2f'  %
                      (os.getpid(), model_id, context.description, weight_seed, pop_name,
                       this_fraction_active_units_threshold))
                sys.stdout.flush()
                return dict()

    return orig_features_dict


def filter_features_multiple_instances(features_dict_list, current_features, model_id=None, export=False, plot=False):
    """

    :param features_dict_list: dict
    :param current_features: dict
    :param model_id: str
    :param export:
    :param plot: bool
    :return: dict
    """

    final_features_dict = {'sparsity_loss': [],
                           'discriminability_loss': [],
                           'selectivity_loss': [],
                           'fraction_active_patterns_loss': [],
                           'fraction_active_units_loss': [],
                           'sparsity': [],
                           'similarity': [],
                           'selectivity': [],
                           'fraction_active_patterns': [],
                           'fraction_active_units': []}

    for orig_features_dict in features_dict_list:
        sparsity_errors = (context.target_val['sparsity'] -
                           orig_features_dict['sparsity_array']) / context.target_range['sparsity']

        # penalize discriminability of silent patterns
        bad_indexes = np.where(np.isnan(orig_features_dict['similarity_array']))
        orig_features_dict['similarity_array'][bad_indexes] = 1.
        discriminability_errors = (context.target_val['similarity'] -
                                   orig_features_dict['similarity_array']) / context.target_range['similarity']

        selectivity_errors = (context.target_val['selectivity'] -
                              orig_features_dict['selectivity_array']) / context.target_range['selectivity']
        fraction_active_patterns_error = (context.target_val['fraction_active_patterns'] -
                                 orig_features_dict['fraction_active_patterns']) / \
                                context.target_range['fraction_active_patterns']
        fraction_active_units_error = (context.target_val['fraction_active_units'] -
                                          orig_features_dict['fraction_active_units']) / \
                                         context.target_range['fraction_active_units']

        final_features_dict['sparsity_loss'].append(np.sum(sparsity_errors ** 2))
        final_features_dict['discriminability_loss'].append(np.nansum(discriminability_errors ** 2))
        final_features_dict['selectivity_loss'].append(np.sum(selectivity_errors ** 2))
        final_features_dict['fraction_active_patterns_loss'].append(fraction_active_patterns_error ** 2)
        final_features_dict['fraction_active_units_loss'].append(fraction_active_units_error ** 2)
        final_features_dict['sparsity'].append(np.mean(orig_features_dict['sparsity_array']))
        final_features_dict['similarity'].append(np.nanmean(orig_features_dict['similarity_array']))
        final_features_dict['selectivity'].append(np.mean(orig_features_dict['selectivity_array']))
        final_features_dict['fraction_active_patterns'].append(orig_features_dict['fraction_active_patterns'])
        final_features_dict['fraction_active_units'].append(orig_features_dict['fraction_active_units'])

    if context.debug:
        print(final_features_dict)
        sys.stdout.flush()
        context.update(locals())

    return final_features_dict


def get_objectives_multiple_instances(final_features_dict, model_id=None, export=False, plot=False):
    """
    Compute loss function.
    :param final_features_dict: dict
    :param model_id: str
    :param export: bool
    :param plot: bool
    :return: tuple of dict
    """
    objectives_dict = {}
    summary_features_dict = {}
    for feature_name in final_features_dict:
        if 'loss' in feature_name:
            objectives_dict[feature_name] = np.mean(final_features_dict[feature_name])
        else:
            summary_features_dict[feature_name] = np.mean(final_features_dict[feature_name])

    return summary_features_dict, objectives_dict


#############################################################################
''' 
Main function:
- used to run a single instance of a model from the command line.
- requires a yaml file containing the relevant model parameters.

Example command to run from terminal:
python optimize_dynamic_model.py --config_file_path=config/simulate_config_2_FF_Inh.yaml --plot 
'''

@click.command()
@click.option("--config_file_path", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--export_file_name", type=str, default=None)
@click.option("--data_dir", type=click.Path(exists=True, file_okay=False, dir_okay=True), default='data')
@click.option("--plot", is_flag=True)
@click.option("--export", is_flag=True)

def main(config_file_path, export_file_name, data_dir, plot, export):
    """
    Given model configuration parameters, build a network, run a simulation and analyze the output.
    Optionally can generate summary plots and/or export data to an hdf5 file.
    :param config_file_path: str name of .yaml file containing configuration parameters
    :param export_file_name: str; hdf5 file name for data export
    :param data_dir: str (path); directory to export data
    :param plot: bool; whether to generate plots
    :param export: bool; whether to export data to hdf5
    """
    start_time = time.time()
    parameter_dict = read_from_yaml(config_file_path)
    description = parameter_dict['description']

    num_units_dict = parameter_dict['num_units_dict']
    num_input_units = num_units_dict['Input']

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
    t = np.arange(0., duration + dt / 2., dt)
    channel_conductance_dynamics_dict, net_current_dynamics_dict, cell_voltage_dynamics_dict, \
    network_activity_dynamics_dict = \
        get_network_dynamics_dicts(t, sorted_input_patterns, num_units_dict, synapse_tau_dict, cell_tau_dict,
                                   weight_dict, weight_config_dict, activation_function_dict, synaptic_reversal_dict)

    # Analyze the average network dynamics in a time window
    network_activity_dict = slice_network_activity_dynamics_dict(network_activity_dynamics_dict, t,
                                                                 time_point=time_point)

    sparsity_dict, similarity_matrix_dict, selectivity_dict, fraction_active_patterns_dict, \
    fraction_active_units_dict = analyze_slice(network_activity_dict)

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
                                weight_config_dict, network_activity_dict)

        export_file_path = export_file_path[:-5] + '_dynamics.hdf5'
        export_dynamic_model_data(export_file_path, description, weight_seed, model_config_dict, num_units_dict,
                                  activation_function_dict, weight_config_dict, weight_dict, cell_tau_dict,
                                  synapse_tau_dict, channel_conductance_dynamics_dict, net_current_dynamics_dict,
                                  cell_voltage_dynamics_dict, network_activity_dynamics_dict)

    if plot:
        plot_dynamics(t, median_sparsity_dynamics_dict, median_similarity_dynamics_dict, mean_selectivity_dynamics_dict,
                      fraction_nonzero_response_dynamics_dict, description)

        plot_model_summary(network_activity_dict, sparsity_dict, similarity_matrix_dict, selectivity_dict, description)

    print('Simulation took %.1f s' % (time.time() - start_time))

    plt.show()  # this forces all plots generated with fig.show() to wait for the user to close them before exiting python

    context.update(locals())


if __name__ == '__main__':
    # this extra flag stops the click module from forcing system exit when python is in interactive mode
    main(standalone_mode=False)
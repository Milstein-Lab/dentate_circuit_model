# This is an executable script that duals as an importable module
# It has a command line interface via the python module "click"

# Below are functions used to complete the challenge from Tutorial 5.
# The goal was to compare sparsity and discriminability in networks with only FF vs. both FF and FB inhibition.

# Now that model output is computed over a time series by scipy.integrate.solve_ivp, we need versions of our analysis
# and plotting scripts that can:
# 1) Plot sparsity and discriminability for a single time slice of the network activity
# and
# 2) Plot mean sparsity over time


import click
import datetime
import h5py
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from collections.abc import Iterable
from copy import deepcopy
from scipy.integrate import solve_ivp
from nested.utils import read_from_yaml, Context, param_array_to_dict
from distutils.util import strtobool
import os, time

import warnings
warnings.filterwarnings("ignore")


context = Context()


# we'll need some functions that we have been storing in our Jupyter notebook:
def recursive_append_binary_input_patterns(n, index=None, input_pattern_list=None):
    """
    Recursively copy and extend all patterns in the current pattern list to include all binary combinations of length n.
    :param n: int
    :param index: int
    :param input_pattern_list: list of lists of int
    :return: list of lists
    """
    if input_pattern_list is None:
        input_pattern_list = []
        index = 0
        # If the current list of patterns is empty, generate the first two patterns with length 1
        input_pattern_list.append([0])
        input_pattern_list.append([1])
    else:
        # Otherwise, duplicate all previous input patterns of length column - 1, and append either 0 or 1
        prev_num_patterns = len(input_pattern_list)
        for i in range(prev_num_patterns):
            pattern0 = input_pattern_list[i]
            pattern1 = list(pattern0)
            # This modifies pattern0, which is already contained in the input_pattern_list
            pattern0.append(0)
            pattern1.append(1)
            # This modifies pattern1, which needs to be appended to the input_pattern_list
            input_pattern_list.append(pattern1)
    if index >= n - 1:
        return input_pattern_list
    else:
        return recursive_append_binary_input_patterns(n, index + 1, input_pattern_list)


def get_binary_input_patterns(n, sort=True, plot=False):
    """
    Return a 2D numpy array with 2 ** n rows and n columns containing all possible patterns (rows) comprised of n units
    (columns) that can either be on (0) or off (1).
    :param n: int; number of units
    :param sort: bool; whether to sort input patterns by the summed activity of the inputs
    :param plot: bool; whether to plot
    :return: 2d array
    """
    input_pattern_list = recursive_append_binary_input_patterns(n)
    input_pattern_array = np.array(input_pattern_list)
    if sort:
        summed_input_activities = np.sum(input_pattern_array, axis=1)
        sorted_indexes = np.argsort(summed_input_activities)
        input_pattern_array = input_pattern_array[sorted_indexes]
        summed_input_activities = summed_input_activities[sorted_indexes]
    if plot:
        fig = plt.figure(figsize=(7, 5))
        ax1 = plt.subplot2grid((1, 6), (0, 0), colspan=5, rowspan=1)
        ax1.imshow(input_pattern_array, cmap='binary', aspect='auto')
        ax1.set_title('Input unit activity')
        ax1.set_xlabel('Input unit ID')
        ax1.set_ylabel('Input pattern ID')
        ax2 = plt.subplot2grid((1, 6), (0, 5), colspan=1, rowspan=1)
        ax2.imshow(np.atleast_2d(summed_input_activities).T, cmap='binary', aspect='auto')
        ax2.yaxis.set_label_position("right")
        ax2.set_ylabel('Summed unit activity', rotation=-90., labelpad=30)
        ax2.set_xticks([])
        ax2.set_yticks([])
        fig.tight_layout(w_pad=4.)
        fig.show()

    return input_pattern_array


def act_funct_args(population, activation_function_dict):
    """
    Gets custom parameters for the given activation function if they are provided
    :param population: string
    :param activation_function_dict: dictionary
    :return: dictionary
    """
    if 'Arguments' in activation_function_dict[population]:
        this_kwargs = activation_function_dict[population]['Arguments']
    else:
        this_kwargs = {}
    return this_kwargs


def identity_activation(x):
    return x


def piecewise_linear_activation(weighted_input, peak_output=3., peak_input=7., threshold=0.):
    """
    Output is zero below a threshold, then increases linearly for inputs up to the specified maximum values for inputs
    and output.
    :param weighted_input: array of float
    :param peak_output: float
    :param peak_input: float
    :param threshold: float
    :return: array of float
    """

    slope = peak_output / (peak_input - threshold)

    input_above_threshold = np.maximum(0., weighted_input-threshold)
    output = slope * np.minimum(input_above_threshold, peak_input-threshold)

    return output


def get_callable_from_str(func_name):
    """
    Look for a callable function with the specified name in the global namespace.
    :param func_name: str
    :return: callable
    """
    if func_name in globals():
        func = globals()[func_name]
    elif hasattr(sys.modules[__name__], func_name):
        func = getattr(sys.modules[__name__], func_name)
    else:
        raise RuntimeError('get_callable_from_str: %s not found' % func_name)
    if callable(func):
        return func
    else:
        raise RuntimeError('get_callable_from_str: %s not found' % func_name)


def get_d_syn_current_dt_array(syn_current, pre_activity, weights, synapse_tau, synapse_scalar):
    """
    Compute the rates of change of all synapses from all the units in a single presynaptic population to all the
    units in a single postsynaptic population. Initial currents are provided as a 2D matrix. Initial presynaptic
    activity is provided as a 1D array. Weights are provided as a 2D matrix.
    :param syn_current: array of float (number of presynaptic units, number of postsynaptic units)
    :param pre_activity: array of float (number of presynaptic units)
    :param weights: array of float (number of presynaptic units, number of postsynaptic units)
    :param synapse_tau: float (seconds)
    :param synapse_scalar: float
    :return: array of float (number of presynaptic units, number of postsynaptic units)
    """
    # repeat the array of presynaptic activities along columns to match the shape of the weights and currents
    d_syn_current_dt_array = \
        -syn_current / synapse_tau + weights * pre_activity[:, None] * synapse_scalar
    return d_syn_current_dt_array


def get_d_cell_voltage_dt_array(cell_voltage, net_current, cell_tau, input_resistance=1.):
    """
    Computes the rates of change of cellular voltage in all units of a single population. Initial cell voltages are
    provided as a 1D array. The summed initial synaptic currents are provided as a 1D array.
    :param cell_voltage: array of float (num units in population)
    :param net_current: array of float (num units in population)
    :param cell_tau: float (seconds)
    :param input_resistance: float
    :return: array of float (num units in population)
    """
    d_cell_voltage_dt_array = (-cell_voltage + input_resistance * net_current) / cell_tau
    return d_cell_voltage_dt_array


def get_d_conductance_dt_array(channel_conductance, pre_activity, rise_tau, decay_tau):
    d_conductance_dt_array = -channel_conductance / decay_tau + \
                             np.maximum(0., pre_activity[:, None] - channel_conductance) / rise_tau
    return d_conductance_dt_array


def get_net_current(weights, channel_conductances, cell_voltage, reversal_potential):
    net_current_array = ((weights * channel_conductances) * (reversal_potential - cell_voltage))
    net_current_array = np.sum(net_current_array, axis=0)
    return net_current_array


def get_d_network_intermediates_dt_dicts(num_units_dict, synapse_tau_dict, cell_tau_dict, weight_dict,
                                         weight_config_dict, synaptic_reversal_dict, channel_conductance_dict,
                                         cell_voltage_dict, network_activity_dict):
    """
    Computes rates of change of all synaptic currents and all cell voltages for all populations in a network.
    :param num_units_dict: dict: {'population': int (number of units in this population)}
    :param synapse_tau_dict: nested dict:
        {'postsynaptic population label':
            {'presynaptic population label':
                float (seconds)
            }
        }
    :param cell_tau_dict:
        {'population label':
            float (seconds)
        }
    :param weight_dict: nested dict:
        {'postsynaptic population label':
            {'presynaptic population label':
                2d array of float (number of presynaptic units, number of postsynaptic units)
            }
        }
    :param weight_config_dict: nested dict:
        {'postsynaptic population label':
            {'resynaptic population label':
                {'distribution: string
    :param syn_current_dict:
        {'post population label':
            {'pre population label':
                2d array of float (number of presynaptic units, number of postsynaptic units)
            }
        }
    :param cell_voltage_dict:
        {'population label':
            1d array of float (number of units)
        }
    :param network_activity_dict:
        {'population label':
            1d array of float (number of units)
        }
    :return: tuple of dict: (d_syn_current_dt_dict, d_cell_voltage_dt_dict)
    """
    d_syn_current_dt_dict = {}
    d_cell_voltage_dt_dict = {}
    d_conductance_dt_dict = {}

    for post_population in weight_dict:  # get the change in synaptic currents for every connection
        d_conductance_dt_dict[post_population] = {}
        this_cell_tau = cell_tau_dict[post_population]
        this_net_current = np.zeros_like(network_activity_dict[post_population])
        this_cell_voltage = cell_voltage_dict[post_population]
        for pre_population in weight_dict[post_population]:

            this_decay_tau = synapse_tau_dict[post_population][pre_population]['decay']
            this_rise_tau = synapse_tau_dict[post_population][pre_population]['rise']

            this_channel_conductance = channel_conductance_dict[post_population][pre_population]

            this_pre_activity = network_activity_dict[pre_population]
            d_conductance_dt_dict[post_population][pre_population] = \
                get_d_conductance_dt_array(this_channel_conductance, this_pre_activity, this_rise_tau, this_decay_tau)

            this_weights = weight_dict[post_population][pre_population]

            this_connection_type = weight_config_dict[post_population][pre_population]['connection_type']
            this_reversal_potential = synaptic_reversal_dict[this_connection_type]

            # TODO(done): np.sum should be inside get_net_current
            this_net_current += get_net_current(this_weights, channel_conductance_dict[post_population][pre_population],
                                                       this_cell_voltage, this_reversal_potential)
            #this_net_current += np.sum(syn_current_dict[post_population][pre_population], axis=0)
        d_cell_voltage_dt_dict[post_population] = \
            get_d_cell_voltage_dt_array(this_cell_voltage, this_net_current, this_cell_tau)

    return d_conductance_dt_dict, d_cell_voltage_dt_dict


def nested_dicts_to_flat_state_list(channel_conductance_dict, cell_voltage_dict):
    """
    Given nested dictionaries of synaptic currents and cell_voltages, return a flat list of state variables.
    Also return a legend that can be used to re-construct the original nested dictionaries. This is in the form
    of a nested dictionary of indexes into the flat list.
    :param cell_voltage_dict: dict of voltages by population; {pop_name: 1d array of float}
    :param syn_current_dict: dict of synaptic currents by projection; {post_pop_name: {pre_pop_name: 2d array of float}}
    :return: tuple; (list of float, nested dict: tuple of int indexes)
    """
    legend = dict()
    state_list = []
    legend['cell_voltage'] = dict()
    start = 0
    legend['channel_conductance'] = dict()
    for post_population in sorted(list(channel_conductance_dict.keys())):
        legend['channel_conductance'][post_population] = {}
        for pre_population in sorted(list(channel_conductance_dict[post_population].keys())):
            state_list.extend(np.ravel(channel_conductance_dict[post_population][pre_population]))
            end = len(state_list)
            legend['channel_conductance'][post_population][pre_population] = (start, end)
            start = end

    for population in sorted(list(cell_voltage_dict.keys())):
        state_list.extend(cell_voltage_dict[population])
        end = len(state_list)
        legend['cell_voltage'][population] = (start, end)
        start = end
    return state_list, legend


def flat_state_list_to_nested_dicts(state_list, legend, num_units_dict):
    """
    Given a flat list of state variables, use the provided legend to construct nested dictionaries of synaptic currents
    and cell_voltages. The legend is in the form of a nested dictionary of indexes into the flat list.
    :param state_list: list of float
    :param legend: nested dict: tuple of of int indexes
    :param num_units_dict: dict of int
    :return: tuple; nested dicts of states by population
    """
    channel_conductance_dict = dict()
    cell_voltage_dict = dict()

    for post_population in sorted(list(legend['channel_conductance'].keys())):
        channel_conductance_dict[post_population] = {}
        for pre_population in sorted(list(legend['channel_conductance'][post_population].keys())):
            start = legend['channel_conductance'][post_population][pre_population][0]
            end = legend['channel_conductance'][post_population][pre_population][1]
            this_state_array = np.array(state_list[start:end]).reshape(
                (num_units_dict[pre_population], num_units_dict[post_population]))
            channel_conductance_dict[post_population][pre_population] = this_state_array

    for population in sorted(list(legend['cell_voltage'].keys())):
        start = legend['cell_voltage'][population][0]
        end = legend['cell_voltage'][population][1]
        cell_voltage_dict[population] = np.array(state_list[start:end])



    return channel_conductance_dict, cell_voltage_dict


def simulate_network_dynamics(t, state_list, legend, input_pattern, num_units_dict, synapse_tau_dict, cell_tau_dict,
                              weight_dict, weight_config_dict,  activation_function_dict, synaptic_reversal_dict):

    """
    Called by scipy.integrate.solve_ivp to compute the rates of change of all network state variables given a flat list
    of initial state values and a pattern of activity in a population of inputs.
    :param synaptic_reversal_dict:
    :param t: float; time point (seconds)
    :param state_list: list of float; flat list of intermediate network variables (synaptic currents and cell voltages)
    :param legend: nested dict: tuple of of int indexes
    :param input_pattern: array of float (num units in Input population)
    :param num_units_dict: dict: {'population': int (number of units in this population)}
    :param synapse_tau_dict: nested dict:
        {'postsynaptic population label':
            {'presynaptic population label':
                float (seconds)
            }
        }
    :param cell_tau_dict:
        {'population label':
            float (seconds)
        }
    :param weight_dict: nested dict:
        {'postsynaptic population label':
            {'presynaptic population label':
                2d array of float (number of presynaptic units, number of postsynaptic units)
            }
        }
    :param activation_function_dict: dict:
        {'population': callable (function to call to convert weighted input to output activity for this population)}
        }
    :return: list of float; (time derivatives of states variables)
    """

    channel_conductance_dict, cell_voltage_dict = flat_state_list_to_nested_dicts(state_list, legend, num_units_dict)

    network_activity_dict = {}
    network_activity_dict['Input'] = np.copy(input_pattern)



    for population in cell_voltage_dict:

        this_activation_function = get_callable_from_str(activation_function_dict[population]['Name'])
        # TODO(done): may want to have a helper function parse this dictionary
        this_kwargs = act_funct_args(population, activation_function_dict)

        network_activity_dict[population] = this_activation_function(cell_voltage_dict[population],
                                                                     **this_kwargs)

    d_conductance_dt_dict, d_cell_voltage_dt_dict = \
        get_d_network_intermediates_dt_dicts(num_units_dict, synapse_tau_dict, cell_tau_dict, weight_dict,
                                             weight_config_dict, synaptic_reversal_dict, channel_conductance_dict,
                                             cell_voltage_dict, network_activity_dict)

    d_state_dt_list, legend = nested_dicts_to_flat_state_list(d_conductance_dt_dict, d_cell_voltage_dt_dict)

    return d_state_dt_list



def state_dynamics_to_nested_dicts(state_dynamics, legend, input_pattern, num_units_dict, activation_function_dict,
                                   weight_dict, cell_voltage_dict, weight_config_dict, synaptic_reversal_dict):
    """
    The output of scipy.integrate.solve_ivp is a 2D array containing the values of all network state variables (rows)
    over time (columns). This function uses the provided legend to construct nested dictionaries of network
    intermediates and cell activities. The legend is in the form of a nested dictionary of indexes into the rows of the
    state matrix.
    :param state_dynamics: 2d array of float (num state variables, num time points)
    :param legend: nested dict of int indexes
    :param input_pattern: array of float (num units in Input population)
    :param num_units_dict: dict: {'population': int (number of units in this population)}
    :param activation_function_dict: dict:
        {'population': callable (function to call to convert weighted input to output activity for this population)}
        }
    :return: tuple of nested dict
    """
    len_t = state_dynamics.shape[1]

    #syn_current_dynamics_dict = {}
    net_current_dynamics_dict = {}
    cell_voltage_dynamics_dict = {}
    network_activity_dynamics_dict = {}
    channel_conductance_dynamics_dict = {}

    # fancy way to copy static input pattern across additional dimension of time
    network_activity_dynamics_dict['Input'] = np.broadcast_to(input_pattern[..., None], input_pattern.shape + (len_t,))

    for post_population in legend['channel_conductance']:
        channel_conductance_dynamics_dict[post_population] = {}
        net_current_dynamics_dict[post_population] = np.zeros((num_units_dict[post_population], len_t))
        cell_voltage_dynamics_dict[post_population] = np.empty((num_units_dict[post_population], len_t))
        network_activity_dynamics_dict[post_population] = np.empty((num_units_dict[post_population], len_t))
        for pre_population in legend['channel_conductance'][post_population]:
            channel_conductance_dynamics_dict[post_population][pre_population] = \
                np.empty((num_units_dict[pre_population], num_units_dict[post_population], len_t))

    for i in range(len_t):
        channel_conductance_dict, cell_voltage_dict = \
            flat_state_list_to_nested_dicts(state_dynamics[:,i], legend, num_units_dict)
        for post_population in channel_conductance_dict:
            for pre_population in channel_conductance_dict[post_population]:
                channel_conductance_dynamics_dict[post_population][pre_population][:,:,i] = \
                    channel_conductance_dict[post_population][pre_population]
                this_weights = weight_dict[post_population][pre_population]
                this_cell_voltage = cell_voltage_dict[post_population]
                this_connection_type = weight_config_dict[post_population][pre_population]['connection_type']
                this_reversal_potential = synaptic_reversal_dict[this_connection_type]

                # TODO: np.sum should happen inside get_net_current
                net_current_dynamics_dict[post_population][:,i] += \
                    np.sum(get_net_current(this_weights, channel_conductance_dict[post_population][pre_population],
                                           this_cell_voltage, this_reversal_potential), axis=0)
        for population in cell_voltage_dict:
            cell_voltage_dynamics_dict[population][:,i] = cell_voltage_dict[population]
            this_activation_function = get_callable_from_str(activation_function_dict[population]['Name'])
            this_kwargs = act_funct_args(population, activation_function_dict)
            network_activity_dynamics_dict[population][:,i] = \
                this_activation_function(cell_voltage_dict[population], **this_kwargs)

    return channel_conductance_dynamics_dict, net_current_dynamics_dict, cell_voltage_dynamics_dict, \
           network_activity_dynamics_dict


def compute_network_activity_dynamics(t, input_pattern, num_units_dict, synapse_tau_dict, cell_tau_dict, weight_dict,
                                      weight_config_dict,
                                      activation_function_dict, synaptic_reversal_dict):
    """
    Use scipy.integrate.solve_ivp to calculate network intermediates and activites over time, in response to a single,
    static input pattern.
    :param t: array of float
    :param input_pattern: array of float (num units in Input population)
    :param num_units_dict: dict: {'population': int (number of units in each population)}
    :param synapse_tau_dict: dict of dicts:
        {'postsynaptic population label':
            {'presynaptic population label': float (synaptic time constant for each connection)}}
    :param cell_tau_dict: dict: {'population label': float (voltage time constant for each population)}
    :param weight_dict: nested dict:
        {'postsynaptic population label':
            {'presynaptic population label': 2d array of float
                (number of presynaptic units, number of postsynaptic units)
            }
        }
    :param activation_function_dict: dict:
        {'population': callable (function to call to convert weighted input to output activity for this population)}
        }
    :param synaptic_reversal_dict:
    :return: tuple of nested dict
    """

    # Initialize nested dictionaries to contain network intermediates for one time step
    # in response to one input pattern
    cell_voltage_dict = {}
    network_activity_dict = {}
    channel_conductance_dict = {}

    network_activity_dict['Input'] = np.copy(input_pattern)

    for post_population in weight_dict:
        channel_conductance_dict[post_population] = {}
        for pre_population in weight_dict[post_population]:
            channel_conductance_dict[post_population][pre_population] = np.zeros(
                (num_units_dict[pre_population], num_units_dict[post_population]))

        cell_voltage_dict[post_population] = np.zeros(num_units_dict[post_population])
        network_activity_dict[post_population] = np.zeros(num_units_dict[post_population])

    initial_state_list, legend = nested_dicts_to_flat_state_list(channel_conductance_dict, cell_voltage_dict)

    sol = solve_ivp(simulate_network_dynamics, t_span=(t[0], t[-1]), y0=initial_state_list, t_eval=t,
                    args=(legend, input_pattern, num_units_dict, synapse_tau_dict, cell_tau_dict,
                          weight_dict, weight_config_dict, activation_function_dict, synaptic_reversal_dict))

    channel_conductance_dynamics_dict, net_current_dynamics_dict, cell_voltage_dynamics_dict, \
    network_activity_dynamics_dict = state_dynamics_to_nested_dicts(sol.y, legend, input_pattern, num_units_dict,
                                                                    activation_function_dict, weight_dict,
                                                                    cell_voltage_dict, weight_config_dict,
                                                                    synaptic_reversal_dict)

    return channel_conductance_dynamics_dict, net_current_dynamics_dict, cell_voltage_dynamics_dict, \
        network_activity_dynamics_dict


def get_network_dynamics_dicts(t, input_patterns, num_units_dict, synapse_tau_dict, cell_tau_dict, weight_dict,
                               weight_config_dict,
                               activation_function_dict, synaptic_reversal_dict):
    """
    Use scipy.integrate.solve_ivp to calculate network intermediates and activites over time, in response to a set of
    input patterns.
    static input pattern.
    :param t: array of float
    :param input_patterns: 2D array of float (num input patterns, num units in Input population)
    :param num_units_dict: dict: {'population': int (number of units in each population)}
    :param synapse_tau_dict: dict of dicts:
        {'postsynaptic population label':
            {'presynaptic population label': float (synaptic time constant for each connection)}}
    :param cell_tau_dict: dict: {'population label': float (voltage time constant for each population)}
    :param weight_dict: nested dict:
        {'postsynaptic population label':
            {'presynaptic population label': 2d array of float
                (number of presynaptic units, number of postsynaptic units)
            }
        }
    :param activation_function_dict: dict:
        {'population': callable (function to call to convert weighted input to output activity for this population)}
        }
    :param synaptic_reversal_dict:
    :return: tuple of nested dict
    """

    # Initialize nested dictionaries to contain network intermediates in response to a set of input patterns across
    # all time steps
    channel_conductance_dynamics_dict = {}
    net_current_dynamics_dict = {}
    cell_voltage_dynamics_dict = {}
    network_activity_dynamics_dict = {}

    for population in num_units_dict:
        network_activity_dynamics_dict[population] = np.empty((len(input_patterns), num_units_dict[population], len(t)))

    for post_population in weight_dict:
        channel_conductance_dynamics_dict[post_population] = {}
        for pre_population in weight_dict[post_population]:
            channel_conductance_dynamics_dict[post_population][pre_population] = \
                np.empty((len(input_patterns), num_units_dict[pre_population], num_units_dict[post_population], len(t)))

        net_current_dynamics_dict[post_population] = \
            np.empty((len(input_patterns), num_units_dict[post_population], len(t)))


        cell_voltage_dynamics_dict[post_population] = \
            np.empty((len(input_patterns), num_units_dict[post_population], len(t)))

    for pattern_index in range(len(input_patterns)):
        this_input_pattern = input_patterns[pattern_index]
        this_channel_conductance_dynamics_dict, this_net_current_dynamics_dict, this_cell_voltage_dynamics_dict, \
        this_network_activity_dynamics_dict = \
            compute_network_activity_dynamics(t, this_input_pattern, num_units_dict, synapse_tau_dict, cell_tau_dict,
                                              weight_dict, weight_config_dict, activation_function_dict,
                                              synaptic_reversal_dict)

        for population in this_network_activity_dynamics_dict:
            network_activity_dynamics_dict[population][pattern_index, :, :] = \
                this_network_activity_dynamics_dict[population]

        for post_population in this_channel_conductance_dynamics_dict:
            for pre_population in this_channel_conductance_dynamics_dict[post_population]:
                channel_conductance_dynamics_dict[post_population][pre_population][pattern_index, :, :, :] = \
                    this_channel_conductance_dynamics_dict[post_population][pre_population]
            net_current_dynamics_dict[post_population][pattern_index, :, :] = \
                this_net_current_dynamics_dict[post_population]
            cell_voltage_dynamics_dict[post_population][pattern_index, :, :] = \
                this_cell_voltage_dynamics_dict[post_population]

    return channel_conductance_dynamics_dict, net_current_dynamics_dict, cell_voltage_dynamics_dict, \
           network_activity_dynamics_dict


def slice_network_activity_dynamics_dict(network_activity_dynamics_dict, t, time_point):
    """
    Given network activity dynamics across a set of input patterns over all time points, return network activity across
    all input patterns at the time point specified.
    :param network_activity_dynamics_dict: dict of 3d array of float;
        {'population label':
            3d array of float (number of input patterns, number of units in this population, number of time points)
        }
    :param t: array of float
    :param time_point: float
    :return: dict of 2d array of float;
        {'population label':
            2d array of float (number of input patterns, number of units in this population)
        }
    """
    network_activity_dict = {}

    if isinstance(time_point, (tuple, list)) and len(time_point) == 2:
        t_start_index = np.where(t >= time_point[0])[0][0]
        t_end_index = np.where(t >= time_point[1])[0][0]
        for population in network_activity_dynamics_dict:
            network_activity_dict[population] = \
                np.mean(network_activity_dynamics_dict[population][:, :, t_start_index:t_end_index], axis=2)
    elif isinstance(time_point, (str, int, float)):
        time_point = max(float(time_point), t[-1])
        t_index = np.where(t >= float(time_point))[0][0]
        for population in network_activity_dynamics_dict:
            network_activity_dict[population] = network_activity_dynamics_dict[population][:, :, t_index]
    else:
        raise AssertionError('time_point must be float or length 2 list or tuple')

    return network_activity_dict


def gini_coefficient(x):
    x = np.asarray(x) + 0.0001
    sorted_x = np.sort(x)
    n = len(x)
    cumx = np.cumsum(sorted_x, dtype=float)
    return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n


def analyze_slice(network_activity_dict):
    """
    For each population, for each input pattern, compute sparsity.
    For each population, compare the reponses across all input patterns using cosine similarity. Exclude responses where
    all units in a population are zero.
    :param network_activity_dict: dict:
        {'population label': 2d array of float (number of input patterns, number of units in this population)
        }
    :return: tuple of dict:
        sparsity_dict: dict:
            {'population label': 1d array of float (number of input patterns),
        similarity_matrix_dict: dict:
            {'population label': 2d array of float
                (number of valid response patterns, number of valid response patterns)
            }
    """

    sparsity_dict = {}
    similarity_matrix_dict = {}
    selectivity_dict = {}
    fraction_active_patterns_dict = {}
    fraction_active_units_dict = {}

    num_patterns = network_activity_dict['Input'].shape[0]

    for population in network_activity_dict:

        # number of nonzero units per pattern
        sparsity_dict[population] = np.count_nonzero(network_activity_dict[population], axis=1)

        # if pop activity is 0, remove this sample from similarity calculation
        invalid_indexes = np.where(sparsity_dict[population] == 0.)[0]
        similarity_matrix_dict[population] = cosine_similarity(network_activity_dict[population])
        similarity_matrix_dict[population][invalid_indexes, :] = np.nan
        similarity_matrix_dict[population][:, invalid_indexes] = np.nan

        invalid_indexes = np.where(sparsity_dict[population] == 0.)[0]
        fraction_active_patterns_dict[population] = 1. - len(invalid_indexes) / num_patterns

        # number of nonzero patterns per unit
        selectivity_dict[population] = np.count_nonzero(network_activity_dict[population], axis=0)
        invalid_indexes = np.where(selectivity_dict[population] == 0.)[0]
        fraction_active_units_dict[population] = 1. - len(invalid_indexes) / num_patterns


    return sparsity_dict, similarity_matrix_dict, selectivity_dict, fraction_active_patterns_dict, \
           fraction_active_units_dict


def analyze_dynamics(network_activity_dynamics_dict):
    """
    For each population, for each input pattern, for each time point, compute sparsity. For each time
    point, return the median activity across input patterns.
    For each population, for each time point, compare the reponses across all input patterns using cosine similarity.
    Exclude responses where all units in a population are zero. For each time point, return the median similarity across
    all valid pairs of response patterns.
    :param network_activity_dynamics_dict: dict:
        {'population label': 3d array of float
            (number of input patterns, number of units in this population, number of timepoints)
        }
    :return: :return: tuple of dict:
        median_sparsity_dynamics_dict: dict:
            {'population label': 1d array of float (number of time points),
        median_similarity_dynamics_dict: dict:
            {'population label': 1d array of float (number of time points)
            }

    """
    sparsity_dynamics_dict = {}
    similarity_dynamics_dict = {}
    selectivity_dynamics_dict = {}
    fraction_nonzero_response_dynamics_dict = {}

    first_activity_dynamics_matrix = next(iter(network_activity_dynamics_dict.values()))
    num_patterns = first_activity_dynamics_matrix.shape[0]
    len_t = first_activity_dynamics_matrix.shape[-1]

    for population in network_activity_dynamics_dict:
        pop_size = network_activity_dynamics_dict[population].shape[1]
        sparsity_dynamics_dict[population] = np.empty([len_t,num_patterns])
        similarity_dynamics_dict[population] = np.empty([len_t,num_patterns, num_patterns])
        selectivity_dynamics_dict[population] = np.empty([len_t,pop_size])
        fraction_nonzero_response_dynamics_dict[population] = np.empty([len_t])

        for i in range(len_t):
            sparsity = np.count_nonzero(network_activity_dynamics_dict[population][:, :, i], axis=1)
            sparsity_dynamics_dict[population][i] = sparsity

            invalid_indexes = np.where(sparsity == 0.)[0]
            fraction_nonzero_response_dynamics_dict[population][i] = 1. - len(invalid_indexes) / num_patterns

            similarity_matrix = cosine_similarity(network_activity_dynamics_dict[population][:, :, i])
            similarity_matrix[invalid_indexes, :] = np.nan
            similarity_matrix[:, invalid_indexes] = np.nan
            similarity_dynamics_dict[population][i] = similarity_matrix

            selectivity = np.count_nonzero(network_activity_dynamics_dict[population][:, :, i], axis=0)
            selectivity_dynamics_dict[population][i] = selectivity

            # selectivity = []
            # for unit in range(pop_size):
            #     unit_activity = network_activity_dynamics_dict[population][:, unit, i]
            #     selectivity.append(gini_coefficient(unit_activity))
            # selectivity_dynamics_dict[population][i] = np.array(selectivity)

    return sparsity_dynamics_dict, similarity_dynamics_dict, \
           selectivity_dynamics_dict, fraction_nonzero_response_dynamics_dict


def analyze_median_dynamics(network_activity_dynamics_dict):
    """
    For each population, for each input pattern, for each time point, compute sparsity. For each time
    point, return the median activity across input patterns.
    For each population, for each time point, compare the reponses across all input patterns using cosine similarity.
    Exclude responses where all units in a population are zero. For each time point, return the median similarity across
    all valid pairs of response patterns.
    :param network_activity_dynamics_dict: dict:
        {'population label': 3d array of float
            (number of input patterns, number of units in this population, number of timepoints)
        }
    :return: :return: tuple of dict:
        median_sparsity_dynamics_dict: dict:
            {'population label': 1d array of float (number of time points),
        median_similarity_dynamics_dict: dict:
            {'population label': 1d array of float (number of time points)
            }

    """
    median_sparsity_dynamics_dict = {}
    median_similarity_dynamics_dict = {}
    mean_selectivity_dynamics_dict = {}
    fraction_nonzero_response_dynamics_dict = {}

    first_activity_dynamics_matrix = next(iter(network_activity_dynamics_dict.values()))
    num_patterns = first_activity_dynamics_matrix.shape[0]
    len_t = first_activity_dynamics_matrix.shape[-1]

    for population in network_activity_dynamics_dict:
        median_sparsity_dynamics_dict[population] = np.empty(len_t)
        median_similarity_dynamics_dict[population] = np.empty(len_t)
        mean_selectivity_dynamics_dict[population] = np.empty(len_t)
        fraction_nonzero_response_dynamics_dict[population] = np.empty(len_t)

        for i in range(len_t):
            sparsity = np.count_nonzero(network_activity_dynamics_dict[population][:, :, i], axis=1)
            median_sparsity_dynamics_dict[population][i] = np.median(sparsity)

            invalid_indexes = np.where(sparsity == 0.)[0]
            fraction_nonzero_response_dynamics_dict[population][i] = 1. - len(invalid_indexes) / num_patterns

            similarity_matrix = cosine_similarity(network_activity_dynamics_dict[population][:, :, i])
            similarity_matrix[invalid_indexes, :] = np.nan
            similarity_matrix[:, invalid_indexes] = np.nan
            median_similarity_dynamics_dict[population][i] = np.nanmedian(similarity_matrix)

            nonzero_idx = np.where(network_activity_dynamics_dict[population][:, :, i] > 0)
            selectivity_nonzero_count = np.unique(nonzero_idx[1], return_counts=True)[1]
            mean_selectivity_dynamics_dict[population][i] = np.mean(selectivity_nonzero_count)

            # # Compute selectivity using Gini coefficient:
            # selectivity = []
            # for unit in range(network_activity_dynamics_dict[population].shape[1]):
            #     unit_activity = network_activity_dynamics_dict[population][:, unit, i]
            #     selectivity.append(gini_coefficient(unit_activity))
            # selectivity_dynamics_dict[population][i] = np.array(selectivity)


    return median_sparsity_dynamics_dict, median_similarity_dynamics_dict, \
           mean_selectivity_dynamics_dict, fraction_nonzero_response_dynamics_dict


# def plot_model_summary_single():
    # copy of plot model summary but makes only 1 figure for each population

def plot_model_summary(network_activity_dict, sparsity_dict, similarity_matrix_dict, selectivity_dict, description=None):
    """
    Generate a panel of plots summarizing the activity of each layer.
    :param network_activity_dict: dict:
        {'population label': 2d array of float (number of input patterns, number of units in this population)
        }
    :param sparsity_dict: dict:
        {'population label': 1d array of float (number of input patterns)
        }
    :param similarity_matrix_dict: dict:
        {'population label': 2d array of float (number of input patterns, number of input patterns)
        }
    :param description: str
    """
    num_of_populations = len(network_activity_dict)

    fig, axes = plt.subplots(5, num_of_populations, figsize=(3 * num_of_populations, 8))
    for i, population in enumerate(network_activity_dict):
        # Show activity heatmap of units for all patterns
        im1 = axes[0, i].imshow(network_activity_dict[population], aspect='auto')
        cbar = plt.colorbar(im1, ax=axes[0, i])
        cbar.ax.set_ylabel('Unit activity', rotation=270, labelpad=20)
        axes[0, i].set_xlabel('Unit ID')
        axes[0, i].set_ylabel('Input pattern ID')
        axes[0, i].set_title('Activity\n%s population' % population)

        # Plot sparsity over patterns
        axes[1, i].scatter(range(len(network_activity_dict[population])), sparsity_dict[population])
        axes[1, i].set_xlabel('Input pattern ID')
        axes[1, i].set_ylabel('Num nonzero units')
        axes[1, i].set_title('Sparsity \n%s population' % population)
        axes[1, i].spines["top"].set_visible(False)
        axes[1, i].spines["right"].set_visible(False)

        # Plot similarity matrix heatmap
        im2 = axes[2, i].imshow(similarity_matrix_dict[population], aspect='auto')
        axes[2, i].set_xlabel('Input pattern ID')
        axes[2, i].set_ylabel('Input pattern ID')
        axes[2, i].set_title('Similarity\n%s population' % population)
        plt.colorbar(im2, ax=axes[2, i])

        # Plot discriminability distribution
        bin_width = 0.05
        num_valid_patterns = len(np.where(sparsity_dict[population] > 0.)[0])
        invalid_indexes = np.isnan(similarity_matrix_dict[population])
        if len(invalid_indexes) < similarity_matrix_dict[population].size:
            hist, edges = np.histogram(similarity_matrix_dict[population][~invalid_indexes],
                                       bins=np.arange(-bin_width / 2., 1 + bin_width, bin_width), density=True)
            axes[3, i].plot(edges[:-1] + bin_width / 2., hist * bin_width,
                            label='%.0f inactive pattern' %
                                  (len(sparsity_dict[population]) - num_valid_patterns))
            axes[3, i].set_xlabel('Cosine similarity')
            axes[3, i].set_ylabel('Probability')
            axes[3, i].set_title('Pairwise similarity distribution\n%s population' % population)
            axes[3, i].legend(loc='best', frameon=False)
            axes[3, i].spines["top"].set_visible(False)
            axes[3, i].spines["right"].set_visible(False)

        #Plot selectivity distribution
        num_nonzero_units = np.count_nonzero(selectivity_dict[population])
        active_units_idx = np.where(selectivity_dict[population]>0)
        if num_nonzero_units > 0:
            max_response = np.max(selectivity_dict[population])
        else:
            max_response = 1.
        bin_width = max_response / 20
        hist, edges = np.histogram(selectivity_dict[population][active_units_idx],
                                   bins=np.arange(-bin_width / 2., max_response + bin_width, bin_width), density=True)
        axes[4, i].plot(edges[:-1] + bin_width / 2., hist * bin_width,
                        label='%.0f inactive units' %
                              (len(selectivity_dict[population]) - num_nonzero_units))
        axes[4, i].set_xlabel('Selectivity (# patterns w/ activity)')
        axes[4, i].set_ylabel('Probability')
        axes[4, i].set_title('Selectivity distribution\n%s population' % population)
        axes[4, i].legend(loc='best', frameon=False)
        axes[4, i].spines["top"].set_visible(False)
        axes[4, i].spines["right"].set_visible(False)


    if description is not None:
        fig.suptitle(description)
    fig.tight_layout(w_pad=3, h_pad=3, rect=(0., 0., 1., 0.98))
    fig.show()


def plot_compare_model_sparsity_and_similarity(sparsity_history_dict, similarity_matrix_history_dict):
    """
    Generate a panel of plots comparing different model configuration.
    :param sparsity_history_dict: nested dict:
        {'model description':
            {'population label': 1d array of float (number of input patterns)}
        }
    :param similarity_matrix_history_dict: nested dict:
        {'model description':
            {'population label': 2d array of float (number of input patterns, number of input patterns)}
        }
    """
    fig, axes = plt.subplots(2, 2, figsize=(7, 6))

    for j, description in enumerate(sparsity_history_dict):
        for i, population in enumerate(['Input', 'Output']):
            axes[0, i].scatter(range(len(sparsity_history_dict[description][population])),
                               sparsity_history_dict[description][population], label=description)

            bin_width = 0.05
            num_valid_patterns = len(np.where(sparsity_history_dict[description][population] > 0.)[0])
            invalid_indexes = np.isnan(similarity_matrix_history_dict[description][population])
            if len(invalid_indexes) < similarity_matrix_history_dict[description][population].size:
                hist, edges = np.histogram(similarity_matrix_history_dict[description][population][~invalid_indexes],
                                           bins=np.arange(-bin_width / 2., 1 + bin_width, bin_width), density=True)
                axes[1, i].plot(edges[:-1] + bin_width / 2., hist * bin_width,
                                label='%.0f%% nonzero' %
                                      (100. * num_valid_patterns /
                                       len(sparsity_history_dict[description][population])))

            if j == 0:
                axes[0, i].set_xlabel('Input pattern ID')
                axes[0, i].set_ylabel('Sparsity')
                axes[0, i].set_title('Sparsity\n%s population' % population)
                axes[0, i].spines["top"].set_visible(False)
                axes[0, i].spines["right"].set_visible(False)

                axes[1, i].set_xlabel('Cosine similarity')
                axes[1, i].set_ylabel('Probability')
                axes[1, i].set_title('Pairwise similarity distribution\n%s population' % population)
                axes[1, i].spines["top"].set_visible(False)
                axes[1, i].spines["right"].set_visible(False)

    axes[0, 0].legend(loc='best', frameon=False)
    axes[1, 1].legend(loc='best', frameon=False)

    fig.tight_layout(w_pad=3, h_pad=3, rect=(0., 0., 1., 0.98))
    fig.show()


def plot_dynamics(t, median_sparsity_dynamics_dict, median_similarity_dynamics_dict,
                  mean_selectivity_dynamics_dict, fraction_nonzero_response_dynamics_dict, description=None):
    """
    :param t: array of float
    :param median_sparsity_dynamics_dict: dict:
            {'population label': 1d array of float (number of time points)
    :param median_similarity_dynamics_dict: dict:
            {'population label': 1d array of float (number of time points)
    :param mean_selectivity_dynamics_dict: dict:
            {'population label': 1d array of float (number of time points)
    :param fraction_nonzero_response_dynamics_dict: dict:
            {'population label': 1d array of int (number of time points)
    :param description: str
    """

    num_of_populations = len(median_sparsity_dynamics_dict)
    fig, axes = plt.subplots(4, num_of_populations, figsize=(3.5 * num_of_populations, 9))

    for i, population in enumerate(median_sparsity_dynamics_dict):
        axes[0, i].plot(t, median_sparsity_dynamics_dict[population])
        axes[0, i].set_xlabel('Time (s)')
        axes[0, i].set_ylabel('Sparsity')
        axes[0, i].set_title('Median sparsity\n%s population' % population)
        axes[0, i].spines["top"].set_visible(False)
        axes[0, i].spines["right"].set_visible(False)

        axes[1, i].plot(t, 100. * fraction_nonzero_response_dynamics_dict[population])
        axes[1, i].set_xlabel('Time (s)')
        axes[1, i].set_ylabel('Nonzero responses\n(% of patterns)')
        axes[1, i].set_title('Nonzero responses\n%s population' % population)
        axes[1, i].spines["top"].set_visible(False)
        axes[1, i].spines["right"].set_visible(False)

        axes[2, i].plot(t, median_similarity_dynamics_dict[population])
        axes[2, i].set_xlabel('Time (s)')
        axes[2, i].set_ylabel('Cosine similarity')
        axes[2, i].set_title('Median pairwise similarity\n%s population' % population)
        axes[2, i].spines["top"].set_visible(False)
        axes[2, i].spines["right"].set_visible(False)

        axes[3,i].plot(t, mean_selectivity_dynamics_dict[population])
        axes[3, i].set_xlabel('Time (s)')
        axes[3, i].set_ylabel('Selectivity')
        axes[3, i].set_title('Mean selectivity \n%s population' % population)
        axes[3, i].spines["top"].set_visible(False)
        axes[3, i].spines["right"].set_visible(False)


    if description is not None:
        fig.suptitle(description)
    fig.tight_layout(w_pad=3, h_pad=3, rect=(0., 0., 1., 0.98))
    fig.show()


def plot_compare_sparsity_and_similarity_dynamics(t, median_sparsity_history_dict,
                                                  median_similarity_dynamics_history_dict,
                                                  fraction_nonzero_response_history_dict):
    """
    Compare sparsity and similarity dynamics across model configurations.
    :param t: array of float
    :param median_sparsity_history_dict: nested dict:
            {'model description':
                {'population label': 1d array of float (number of time points)}
            }
    :param median_similarity_dynamics_history_dict: nested dict:
            {'model description':
                {'population label': 1d array of float (number of time points)}
            }
    :param fraction_nonzero_response_dynamics_dict: nested dict:
            {'model description':
                {'population label': 1d array of int (number of time points)}
            }
    """
    fig, axes = plt.subplots(3, 2, figsize=(7, 9))

    for j, description in enumerate(median_sparsity_history_dict):

        for i, population in enumerate(['Input', 'Output']):
            axes[0, i].plot(t, median_sparsity_history_dict[description][population])
            axes[1, i].plot(t, 100. * fraction_nonzero_response_history_dict[description][population],
                            label=description)
            axes[2, i].plot(t, median_similarity_dynamics_history_dict[description][population])
            if j == 0:
                axes[0, i].set_xlabel('Time (s)')
                axes[0, i].set_ylabel('Sparsity activity')
                axes[0, i].set_title('Median sparsity\n%s population' % population)
                axes[0, i].spines["top"].set_visible(False)
                axes[0, i].spines["right"].set_visible(False)

                axes[1, i].set_xlabel('Time (s)')
                axes[1, i].set_ylabel('Nonzero responses\n(% of patterns)')
                axes[1, i].set_title('Nonzero responses\n%s population' % population)
                axes[1, i].spines["top"].set_visible(False)
                axes[1, i].spines["right"].set_visible(False)

                axes[2, i].set_xlabel('Time (s)')
                axes[2, i].set_ylabel('Cosine similarity')
                axes[2, i].set_title('Median pairwise similarity\n%s population' % population)
                axes[2, i].spines["top"].set_visible(False)
                axes[2, i].spines["right"].set_visible(False)

    axes[1, 0].legend(loc='best', frameon=False)

    fig.tight_layout(w_pad=3, h_pad=3, rect=(0., 0., 1., 0.98))
    fig.show()


def get_weight_dict(num_units_dict, weight_config_dict, seed=None, description=None, plot=False):
    """
    Specified are: 1) the number of units in each cell population, 2) the distribution type for weight of each
    projection, 3) the mean weight magnitude for each connection, and 4) a random seed. Returns a nested dictionary
    containing a weight matrix for each projection.
    :param num_units_dict: dict of int
    :param weight_config_dict: nested dict
    :param seed: int
    :param description: str; title for plot
    :param plot: bool
    :return: nested dict: {post_pop: {pre_pop: ndarray of float} }
    """
    if seed is not None:
        np.random.seed(seed)

    weight_dict = {}
    for post_pop in weight_config_dict:
        # The first layer of keys organizes the postsynaptic populations
        weight_dict[post_pop] = {}
        # The second layer of keys organizes the presynaptic populations
        for pre_pop in weight_config_dict[post_pop]:
            dist_type = weight_config_dict[post_pop][pre_pop]['dist_type']
            mean_weight = weight_config_dict[post_pop][pre_pop]['mean_magnitude']
            if dist_type == 'equal':
                weight_dict[post_pop][pre_pop] = \
                    np.ones((num_units_dict[pre_pop], num_units_dict[post_pop])) * mean_weight
            elif dist_type == 'uniform':
                weight_dict[post_pop][pre_pop] = \
                    np.random.uniform(0, mean_weight * 2., (num_units_dict[pre_pop], num_units_dict[post_pop]))
            elif dist_type == 'normal':
                # A normal distribution has a "full width" of 6 standard deviations.
                # This sample will span from zero to 2 * mean_weight.
                weight_dict[post_pop][pre_pop] = \
                    np.random.normal(mean_weight, mean_weight / 3., (num_units_dict[pre_pop], num_units_dict[post_pop]))
                # enforce that normal weights are either all positive or all negative
                weight_dict[post_pop][pre_pop] = np.maximum(0., weight_dict[post_pop][pre_pop])
            elif dist_type == 'log-normal':
                weight_dict[post_pop][pre_pop] = \
                    np.random.lognormal(size=(num_units_dict[pre_pop], num_units_dict[post_pop]))
                # re-scale the weights to match the target mean weight:
                weight_dict[post_pop][pre_pop] = \
                    mean_weight * weight_dict[post_pop][pre_pop] / np.mean(weight_dict[post_pop][pre_pop])
            else:
                raise ValueError('get_weight_dict: unrecognized synaptic weight distribution type: %s' % dist_type)

            if pre_pop == post_pop: #remove self connection from recurrent networks
                np.fill_diagonal(weight_dict[post_pop][pre_pop],0)
    if plot:
        fig = plt.figure()
        for post_pop in weight_dict:
            for pre_pop in weight_dict[post_pop]:
                dist_type = weight_config_dict[post_pop][pre_pop]['dist_type']
                label = '%s -> %s (%s)' % (pre_pop, post_pop, dist_type)
                hist, edges = np.histogram(weight_dict[post_pop][pre_pop], bins=50, density=True)
                bin_size = edges[1] - edges[0]
                plt.plot(edges[:-1] + bin_size, hist * bin_size, label=label)
        plt.legend(loc='best', frameon=False)
        plt.ylabel('Probability')
        plt.xlabel('Weight')
        plt.suptitle('Synaptic weight distributions')
        if description is not None:
            plt.title(description)
        fig.show()

    return weight_dict


def export_model_slice_data(export_file_path, description, weight_seed, model_config_dict, weight_dict,
                            num_units_dict, activation_function_dict, weight_config_dict, network_activity_dict):
    '''
    Export minimal data file required to generate final figures

    :param export_file_path: str (path); path to hdf5 file
    :param description: str; unique identifier for model configuration, used as key in hdf5 file
    :param weight_seed: int
    :param model_config_dict: nested dict
    :param weight_dict: nested dict of ndarray of float
    :param num_units_dict: dict of int
    :param activation_function_dict: dict of str
    :param weight_config_dict: nested dict
    :param network_activity_dict: dict of 2d arrays of float
    '''
    if description is None:
        raise RuntimeError('export_dynamic_model_data: missing required description (unique string identifier for '
                           'model configuration)')

    # This clause evokes a "Context Manager" and takes care of opening and closing the file so we don't forget
    with h5py.File(export_file_path, 'a') as f:
        if description in f:
            model_group = f[description]
        else:
            model_group = f.create_group(description)

        # save the meta data for this model configuration
        for key, value in model_config_dict.items():
            model_group.attrs[key] = value

        seed_group_name = 'seed:'+str(weight_seed)
        if seed_group_name in f[description]: #don't export if the data already exists
            print("ERROR: Data already exists, new values not saved")
            return
        else:
            model_seed_group = model_group.create_group(seed_group_name)

        group = model_seed_group.create_group('weights')
        for post_pop in weight_dict:
            post_group = group.create_group(post_pop)
            for pre_pop in weight_dict[post_pop]:
                post_group.create_dataset(pre_pop, data=weight_dict[post_pop][pre_pop])
                # save the meta data for the weight configuration of this projection
                for key, value in weight_config_dict[post_pop][pre_pop].items():
                    post_group[pre_pop].attrs[key] = value

        group = model_seed_group.create_group('activity')
        for post_pop in network_activity_dict:
            group.create_dataset(post_pop, data=network_activity_dict[post_pop])
            group[post_pop].attrs['num_units'] = num_units_dict[post_pop]
            if post_pop in activation_function_dict:
                group[post_pop].attrs['activation_func_name'] = activation_function_dict[post_pop]['Name']
                group[post_pop].attrs['activation_func_thresh'] = \
                    activation_function_dict[post_pop]['Arguments']['threshold']
                group[post_pop].attrs['activation_func_peak_in'] = \
                    activation_function_dict[post_pop]['Arguments']['peak_input']
                group[post_pop].attrs['activation_func_peak_out'] = \
                    activation_function_dict[post_pop]['Arguments']['peak_output']


def export_dynamic_model_data(export_file_path, description, weight_seed, model_config_dict, num_units_dict,
                              activation_function_dict, weight_config_dict, weight_dict, cell_tau_dict,
                              synapse_tau_dict, channel_conductance_dynamics_dict, net_current_dynamics_dict,
                              cell_voltage_dynamics_dict, network_activity_dynamics_dict):
    """
    Exports data from a single model configuration to hdf5.
    :param export_file_path: str (path); path to hdf5 file
    :param description: str; unique identifier for model configuration, used as key in hdf5 file
    :param weight_seed: int
    :param model_config_dict: nested dict
    :param num_units_dict: dict of int
    :param activation_function_dict: dict of str
    :param weight_config_dict: nested dict
    :param weight_dict: nested dict of ndarray of float
    :param cell_tau_dict: dict of float
    :param synapse_tau_dict: nested dict of float
    :param channel_conductance_dynamics_dict: nested dict of 4d array of float;
        {'post population label':
            'pre population label':
                4d array of float (number of input patterns, number of units in pre population,
                    number of units in post population, number of time points)
    :param net_current_dynamics_dict: dict of 3d array of float;
        {'population label':
            3d array of float (number of input patterns, number of units in this population, number of time points)
        }
    :param cell_voltage_dynamics_dict: dict of 3d array of float;
        {'population label':
            3d array of float (number of input patterns, number of units in this population, number of time points)
        }
    :param network_activity_dynamics_dict: dict of 3d array of float;
        {'population label':
            3d array of float (number of input patterns, number of units in this population, number of time points)
        }
    """
    if description is None:
        raise RuntimeError('export_dynamic_model_data: missing required description (unique string identifier for '
                           'model configuration)')

    # This clause evokes a "Context Manager" and takes care of opening and closing the file so we don't forget
    with h5py.File(export_file_path, 'a') as f:

        if description in f:
            model_group = f[description]
        else:
            model_group = f.create_group(description)

        # save the meta data for this model configuration
        for key, value in model_config_dict.items():
            model_group.attrs[key] = value

        model_seed_group = model_group.create_group(str(weight_seed))

        group = model_seed_group.create_group('weights')
        for post_pop in weight_dict:
            post_group = group.create_group(post_pop)
            for pre_pop in weight_dict[post_pop]:
                post_group.create_dataset(pre_pop, data=weight_dict[post_pop][pre_pop])
                # save the meta data for the weight configuration of this projection
                for key, value in weight_config_dict[post_pop][pre_pop].items():
                    post_group[pre_pop].attrs[key] = value

        group = model_seed_group.create_group('channel_conductances')
        for post_pop in channel_conductance_dynamics_dict:
            subgroup = group.create_group(post_pop)
            for pre_pop in channel_conductance_dynamics_dict[post_pop]:
                subgroup.create_dataset(pre_pop, data=channel_conductance_dynamics_dict[post_pop][pre_pop])
                subgroup[pre_pop].attrs['synapse_rise_tau'] = synapse_tau_dict[post_pop][pre_pop]['rise']
                subgroup[pre_pop].attrs['synapse_decay_tau'] = synapse_tau_dict[post_pop][pre_pop]['decay']

        group = model_seed_group.create_group('net_currents')
        for post_pop in net_current_dynamics_dict:
            group.create_dataset(post_pop, data=net_current_dynamics_dict[post_pop])

        group = model_seed_group.create_group('cell_voltages')
        for post_pop in cell_voltage_dynamics_dict:
            group.create_dataset(post_pop, data=cell_voltage_dynamics_dict[post_pop])
            group[post_pop].attrs['cell_tau'] = cell_tau_dict[post_pop]

        group = model_seed_group.create_group('activity')
        for post_pop in network_activity_dynamics_dict:
            group.create_dataset(post_pop, data=network_activity_dynamics_dict[post_pop])
            group[post_pop].attrs['num_units'] = num_units_dict[post_pop]
            if post_pop in activation_function_dict:
                group[post_pop].attrs['activation_func_name'] = activation_function_dict[post_pop]['Name']
                group[post_pop].attrs['activation_func_thresh'] = \
                    activation_function_dict[post_pop]['Arguments']['threshold']
                group[post_pop].attrs['activation_func_peak_in'] = \
                    activation_function_dict[post_pop]['Arguments']['peak_input']
                group[post_pop].attrs['activation_func_peak_out'] = \
                    activation_function_dict[post_pop]['Arguments']['peak_output']

    print('export_dynamic_model_data: saved data for model %s to %s' % (description, export_file_path))


def import_dynamic_model_data(data_file_path, description=None):
    """
    Imports model data from specified model configurations stored in the specified hdf5 file into nested dictionaries.
    If description is None, the list of model descriptions found in the file are printed.
    If description is 'all', all models found in the file are loaded and returned.
    If description is a valid str or list of str, only data from those model configurations will be imported and
    returned.
    :param data_file_path: str (path); path to hdf5 file
    :param description: str or list of str; unique identifiers for model configurations, used as keys in hdf5 file
    return: tuple of nested dict
    """
    model_config_history_dict = {}
    num_units_history_dict = {}
    activation_function_history_dict = {}
    weight_config_history_dict = {}
    weight_history_dict = {}
    cell_tau_history_dict = {}
    synapse_tau_history_dict = {}
    syn_current_dynamics_history_dict = {}
    net_current_dynamics_history_dict = {}
    cell_voltage_dynamics_history_dict = {}
    network_activity_dynamics_history_dict = {}
    t_history_dict = {}

    # This clause evokes a "Context Manager" and takes care of opening and closing the file so we don't forget
    with h5py.File(data_file_path, 'r') as f:
        if description is None:
            raise RuntimeError('import_model_data: specify one or more valid model descriptions: %s' % list(f.keys()))
        elif isinstance(description, str):
            if description == 'all':
                description_list = list(f.keys())
            elif description in f:
                description_list = [description]
            else:
                raise RuntimeError('import_model_data: model with description: %s not found in %s' %
                                   (description, data_file_path))
        elif isinstance(description, Iterable):
            description_list = list(description)
            for description in description_list:
                if description not in f:
                    raise RuntimeError('import_model_data: model with description: %s not found in %s' %
                                       (description, data_file_path))
        else:
            raise RuntimeError('import_model_data: specify model description as str or list of str')

        for description in description_list:
            model_config_history_dict[description] = {}
            num_units_history_dict[description] = {}
            activation_function_history_dict[description] = {}
            weight_config_history_dict[description] = {}
            weight_history_dict[description] = {}
            cell_tau_history_dict[description] = {}
            synapse_tau_history_dict[description] = {}
            syn_current_dynamics_history_dict[description] = {}
            net_current_dynamics_history_dict[description] = {}
            cell_voltage_dynamics_history_dict[description] = {}
            network_activity_dynamics_history_dict[description] = {}

            model_group = f[description]
            # load the meta data for this model configuration
            for key, value in model_group.attrs.items():
                model_config_history_dict[description][key] = value
            dt = model_config_history_dict[description]['dt']
            duration = model_config_history_dict[description]['duration']
            t_history_dict[description] = np.arange(0., duration + dt / 2., dt)

            group = model_group['weights']
            for post_pop in group:
                weight_config_history_dict[description][post_pop] = {}
                weight_history_dict[description][post_pop] = {}
                for pre_pop in group[post_pop]:
                    weight_config_history_dict[description][post_pop][pre_pop] = {}
                    # load the meta data for the weight configuration of this projection
                    for key, value in group[post_pop][pre_pop].attrs.items():
                        weight_config_history_dict[description][post_pop][pre_pop][key] = value
                    weight_history_dict[description][post_pop][pre_pop] = group[post_pop][pre_pop][:]

            group = model_group['syn_currents']
            for post_pop in group:
                syn_current_dynamics_history_dict[description][post_pop] = {}
                synapse_tau_history_dict[description][post_pop] = {}
                for pre_pop in group[post_pop]:
                    syn_current_dynamics_history_dict[description][post_pop][pre_pop] = group[post_pop][pre_pop][:]
                    synapse_tau_history_dict[description][post_pop][pre_pop] = \
                        group[post_pop][pre_pop].attrs['synapse_tau']

            group = model_group['net_currents']
            for post_pop in group:
                net_current_dynamics_history_dict[description][post_pop] = group[post_pop][:]

            group = model_group['cell_voltages']
            for post_pop in group:
                cell_voltage_dynamics_history_dict[description][post_pop] = group[post_pop][:]
                cell_tau_history_dict[description][post_pop] = group[post_pop].attrs['cell_tau']

            group = model_group['activity']
            for post_pop in group:
                network_activity_dynamics_history_dict[description][post_pop] = group[post_pop][:]
                num_units_history_dict[description][post_pop] = group[post_pop].attrs['num_units']
                if 'activation_function' in group[post_pop].attrs:
                    activation_function_history_dict[description][post_pop] = \
                        get_callable_from_str(group[post_pop].attrs['activation_function'])

    print('import_dynamic_model_data: loaded data from %s for the following model descriptions: %s' %
          (data_file_path, description_list))

    return model_config_history_dict, num_units_history_dict, activation_function_history_dict, \
           weight_config_history_dict, weight_history_dict, cell_tau_history_dict, synapse_tau_history_dict, \
           syn_current_dynamics_history_dict, net_current_dynamics_history_dict, cell_voltage_dynamics_history_dict, \
           network_activity_dynamics_history_dict, t_history_dict


def read_from_yaml(file_path, Loader=None):
    """
    Import a python dict from .yaml
    :param file_path: str (should end in '.yaml')
    :param Loader: :class:'yaml.Loader'
    :return: dict
    """
    import yaml
    if Loader is None:
        Loader = yaml.FullLoader
    if os.path.isfile(file_path):
        with open(file_path, 'r') as stream:
            data = yaml.load(stream, Loader=Loader)
        return data
    else:
        raise Exception('File: {} does not exist.'.format(file_path))


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

    if 'plot' not in context():
        context.plot = False

    if 'debug' not in context():
        context.debug = False

    if 'export_dynamics' not in context():
        context.export_dynamics = False
    elif isinstance(context.export_dynamics, str):
        context.export_dynamics = bool(strtobool(context.export_dynamics))

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


def compute_features(param_array, model_id=None, export=False):
    """

    :param x: array of float
    :param model_id: int
    :param export: bool
    :return: dict
    """

    return compute_features_multiple_instances(param_array, context.weight_seed, model_id, export)


def get_objectives(orig_features_dict, model_id=None, export=False):
    """
    Compute loss function.
    :param features_dict: dict
    :param model_id: int
    :param export: bool
    :return: tuple of dict
    """

    sparsity_errors = (context.target_val['sparsity'] -
                       orig_features_dict['sparsity_array'])/context.target_range['sparsity']
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


def compute_features_multiple_instances(param_array, weight_seed, model_id=None, export=False):
    """

    :param param_array: array of float
    :param weight_seed: int
    :param model_id: int
    :param export: bool
    :return: dict
    """
    start_time = time.time()

    param_dict = param_array_to_dict(param_array, context.param_names)
    modify_network(param_dict) #update the weight config dict

    weight_seed = int(weight_seed)

    weight_dict = get_weight_dict(context.num_units_dict, context.weight_config_dict, weight_seed,
                                  description=context.description, plot=context.plot)

    channel_conductance_dynamics_dict, net_current_dynamics_dict, cell_voltage_dynamics_dict, \
    network_activity_dynamics_dict = \
        get_network_dynamics_dicts(context.t, context.sorted_input_patterns, context.num_units_dict,
                                   context.synapse_tau_dict, context.cell_tau_dict,
                                   weight_dict, context.weight_config_dict, context.activation_function_dict,
                                   context.synaptic_reversal_dict)

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
                                      context.weight_config_dict,
                                      weight_dict, context.cell_tau_dict, context.synapse_tau_dict,
                                      channel_conductance_dynamics_dict, net_current_dynamics_dict,
                                      cell_voltage_dynamics_dict, network_activity_dynamics_dict)
        else:
            export_model_slice_data(context.temp_output_path, context.description, weight_seed, model_config_dict,
                                    weight_dict, context.num_units_dict, context.activation_function_dict,
                                    context.weight_config_dict, network_activity_dict)

    if context.plot:
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
                print('pid: %i; model_id: %i failed; description: %s, weight_seed: %i; population: %s did not meet'
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
                print('pid: %i; model_id: %i failed; description: %s, weight_seed: %i; population: %s did not meet'
                      ' fraction_active_units_threshold: %.2f'  %
                      (os.getpid(), model_id, context.description, weight_seed, pop_name,
                       this_fraction_active_units_threshold))
                sys.stdout.flush()
                return dict()

    return orig_features_dict


def filter_features_multiple_instances(features_dict_list, current_features, model_id=None, export=False):
    """

    :param features_dict_list:
    :param model_id:
    :param export:
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


def get_objectives_multiple_instances(final_features_dict, model_id=None, export=False):
    """
    Compute loss function.
    :param features_dict: dict
    :param model_id: int
    :param export: bool
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
# Example command to run from terminal:
# python -i simulate_dynamic_model_read_from_yaml.py --config_file_path=../config/optimize_config_2_FF_Inh.yaml

@click.command()
@click.option("--config_file_path", type=click.Path(exists=True, file_okay=True, dir_okay=False))
#Time paramters
@click.option("--dt", type=float, default=0.001)  # sec
@click.option("--duration", type=float, default=0.2)  # sec
@click.option("--time_point", type=float, default=0.2)  # sec
#Other optional arguments
@click.option("--weight_seed", type=int, default=None)
@click.option("--description", type=str, default=None)
@click.option("--export_file_name", type=str, default=None)
@click.option("--data_dir", type=click.Path(exists=True, file_okay=False, dir_okay=True), default='data')
@click.option("--plot", is_flag=True)
@click.option("--export", is_flag=True)
def main(config_file_path, dt, duration, time_point, weight_seed, description, export_file_name, data_dir, plot,
         export):
    """
    Given model configuration parameters, build a network, run a simulation and analyze the output.
    Optionally can generate summary plots and/or export data to an hdf5 file.
    :param config_file_path: str name of .yaml file containing configuration parameters
    :param dt: float; time step (in sec) for simulation of activity dynamics
    :param duration: float; total length (in sec) of simulated activity dynamics
    :param time_point: float; time point to analyze final sparsity and discriminability
    :param weight_seed: int; random seed for random but reproducible weights
    :param description: str; unique identifier for model configuration and data export
    :param export_file_name: str; hdf5 file name for data export
    :param data_dir: str (path); directory to export data
    :param plot: bool; whether to generate plots
    :param export: bool; whether to export data to hdf5
    """
    start_time = time.time()
    parameter_dict = read_from_yaml(config_file_path)
    num_units_dict = parameter_dict['num_units_dict']

    num_input_units = num_units_dict['Input']

    # generate all possible binary input patterns with specified number units in the input layer
    sorted_input_patterns = get_binary_input_patterns(num_input_units, sort=True, plot=plot)

    num_units_dict['Output'] = len(sorted_input_patterns)

    # Let's be explicit about whether we will apply a nonlinear activation function to each population:
    activation_function_dict = parameter_dict['activation_function_dict']

    weight_config_dict = parameter_dict['weight_config_dict']

    cell_tau_dict = parameter_dict['cell_tau_dict']

    synapse_tau_dict = parameter_dict['synapse_tau_dict']

    weight_dict = get_weight_dict(num_units_dict, weight_config_dict, weight_seed, description=description, plot=plot)

    synaptic_reversal_dict = parameter_dict['synaptic_reversal_dict']

    t = np.arange(0., duration + dt / 2., dt)

    channel_conductance_dynamics_dict, net_current_dynamics_dict, cell_voltage_dynamics_dict, \
    network_activity_dynamics_dict = \
        get_network_dynamics_dicts(t, sorted_input_patterns, num_units_dict, synapse_tau_dict, cell_tau_dict,
                                   weight_dict, weight_config_dict, activation_function_dict, synaptic_reversal_dict)

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

        export_model_slice_data(context.export_file_path, context.description, context.weight_seed, model_config_dict,
                                weight_dict, context.num_units_dict, context.activation_function_dict,
                                context.weight_config_dict, network_activity_dict)

        export_dynamic_model_data(export_file_path, description, weight_seed, model_config_dict, num_units_dict,
                                  activation_function_dict, weight_config_dict, weight_dict, cell_tau_dict,
                                  synapse_tau_dict, channel_conductance_dynamics_dict, net_current_dynamics_dict,
                                  cell_voltage_dynamics_dict, network_activity_dynamics_dict)

    if plot:
        plot_model_summary(network_activity_dict, sparsity_dict, similarity_matrix_dict, description)
        plot_sparsity_and_similarity_dynamics(t, median_sparsity_dynamics_dict,
                                              median_similarity_dynamics_dict, fraction_nonzero_response_dynamics_dict,
                                              description)

    # this forces all plots generated with fig.show() to wait for the user to close them before exiting python
    print('Simulation took %.1f s' % (time.time() - start_time))
    plt.show()

    context.update(locals())


if __name__ == '__main__':
    # this extra flag stops the click module from forcing system exit when python is in interactive mode
    main(standalone_mode=False)
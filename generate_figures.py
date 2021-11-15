# python generate_figures.py --data_file_path=data/20211115_exported_dentate_model_data.hdf5 --dynamics_file_path=data/20211115_exported_dentate_model_data_dynamics.hdf5

import click
import numpy as np
from math import ceil
import h5py
from copy import deepcopy
import matplotlib.pyplot as plt
from optimize_dynamic_model import analyze_slice, get_binary_input_patterns

import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

import seaborn as sns
sns.set_style(style='ticks')

def import_slice_data(data_file_path):
    """
    Imports model data from specified model configurations stored in the specified hdf5 file into nested dictionaries.
    If model_seed is None, the list of model model_seeds found in the file are printed.
    If model_seed is 'all', all models found in the file are loaded and returned.
    If model_seed is a valid str or list of str, only data from those model configurations will be imported and
    returned.
    :param data_file_path: str (path); path to hdf5 file
    :param model_seed: str or list of str; unique identifiers for model configurations, used as keys in hdf5 file
    """

    model_config_history_dict = {}
    num_units_history_dict = {}
    activation_function_history_dict = {}
    weight_config_history_dict = {}
    weight_history_dict = {}
    network_activity_history_dict = {}
    sparsity_history_dict = {}
    similarity_matrix_history_dict = {}
    selectivity_history_dict = {}
    fraction_active_patterns_history_dict = {}
    fraction_active_units_history_dict = {}

    # This clause evokes a "Context Manager" and takes care of opening and closing the file so we don't forget
    with h5py.File(data_file_path, 'r') as f:
        description_list = list(f.keys())

        for description in description_list:

            model_config_history_dict[description] = {}
            num_units_history_dict[description] = {}
            activation_function_history_dict[description] = {}
            weight_config_history_dict[description] = {}
            weight_history_dict[description] = {}
            network_activity_history_dict[description] = {}
            sparsity_history_dict[description] = {}
            similarity_matrix_history_dict[description] = {}
            selectivity_history_dict[description] = {}
            fraction_active_patterns_history_dict[description] = {}
            fraction_active_units_history_dict[description] = {}

            model_seed_list = list(f[description].keys())
            for model_seed in model_seed_list:
                model_config_dict = {}
                num_units_dict = {}
                activation_function_dict = {}
                weight_config_dict = {}
                weight_dict = {}
                network_activity_dict = {}

                model_group = f[description][model_seed]
                # load the meta data for this model configuration
                for key, value in model_group.attrs.items():
                    model_config_dict[key] = value

                group = model_group['weights']
                for post_pop in group:
                    weight_dict[post_pop] = {}
                    weight_config_dict[post_pop] = {}
                    for pre_pop in group[post_pop]:
                        weight_dict[post_pop][pre_pop] = group[post_pop][pre_pop][:]
                        weight_config_dict[post_pop][pre_pop] = {}
                        for key, value in group[post_pop][pre_pop].attrs.items():
                            weight_config_dict[post_pop][pre_pop][key] = value

                group = model_group['activity']
                for post_pop in group:
                    network_activity_dict[post_pop] = group[post_pop][:]
                    num_units_dict[post_pop] = group[post_pop].attrs['num_units']
                    if 'activation_function' in group[post_pop].attrs:
                        activation_function_dict[post_pop] = \
                            get_callable_from_str(group[post_pop].attrs['activation_function'])

                sparsity_dict, similarity_matrix_dict, selectivity_dict, \
                    fraction_active_patterns_dict,fraction_active_units_dict = analyze_slice(network_activity_dict)

                model_config_history_dict[description][model_seed] = deepcopy(model_config_dict)
                num_units_history_dict[description][model_seed] = deepcopy(num_units_dict)
                activation_function_history_dict[description][model_seed] = deepcopy(activation_function_dict)
                weight_config_history_dict[description][model_seed] = deepcopy(weight_config_dict)
                weight_history_dict[description][model_seed] = deepcopy(weight_dict)
                network_activity_history_dict[description][model_seed] = deepcopy(network_activity_dict)
                sparsity_history_dict[description][model_seed] = deepcopy(sparsity_dict)
                similarity_matrix_history_dict[description][model_seed] = deepcopy(similarity_matrix_dict)
                selectivity_history_dict[description][model_seed] = deepcopy(selectivity_dict)
                fraction_active_patterns_history_dict[description][model_seed] = deepcopy(fraction_active_patterns_dict)
                fraction_active_units_history_dict[description][model_seed] = deepcopy(fraction_active_units_dict)

        # print('import_model_data: loaded data from {} for the following models: {}, {}'.format(\
        #     data_file_path, description, model_seed_list))


    return  model_config_history_dict, num_units_history_dict, activation_function_history_dict, \
            weight_config_history_dict, weight_history_dict, network_activity_history_dict, \
            sparsity_history_dict, similarity_matrix_history_dict, selectivity_history_dict, \
            fraction_active_patterns_history_dict, fraction_active_units_history_dict


def import_dynamic_activity(data_file_path):
    """
    Imports model data from specified model configurations stored in the specified hdf5 file into nested dictionaries.
     :param data_file_path: str (path); path to hdf5 file
    """

    network_activity_dynamics_history_dict = {}
    with h5py.File(data_file_path, 'r') as f:
        description = list(f.keys())[0]
        network_activity_dynamics_history_dict[description] = {}
        model_seed_list = list(f[description].keys())
        for model_seed in model_seed_list:
            network_activity_dict = {}
            model_group = f[description][model_seed]
            group = model_group['activity']
            for post_pop in group:
                network_activity_dict[post_pop] = group[post_pop][:]

            network_activity_dynamics_history_dict[description][model_seed] = deepcopy(network_activity_dict)

    return  network_activity_dynamics_history_dict


def plot_cumulative_discriminability(similarity_matrix_history_dict):
    cumulative_discriminability = []
    n_bins = 100
    cdf_prob_bins = np.arange(1., n_bins + 1.) / n_bins

    for model_seed in similarity_matrix_history_dict:
        similarity_matrix = similarity_matrix_history_dict[model_seed]['Output']

        # extract all values below diagonal
        similarity_matrix_idx = np.tril_indices_from(similarity_matrix, -1)
        similarity = similarity_matrix[similarity_matrix_idx]

        #remove nan values
        invalid_idx = np.isnan(similarity)
        similarity[invalid_idx] = 1
        # similarity = similarity[~invalid_idx]

        discriminability = 1 - similarity

        discriminability = np.sort(discriminability[:])
        quantiles = [np.quantile(discriminability, pi) for pi in cdf_prob_bins]
        cumulative_discriminability.append(quantiles)

    cumulative_discriminability = np.array(cumulative_discriminability)
    mean_discriminability = np.mean(cumulative_discriminability, axis=0)
    SD = np.std(cumulative_discriminability, axis=0)
    SEM = SD / np.sqrt(cumulative_discriminability.shape[0])

    return mean_discriminability, cdf_prob_bins, SD


def plot_cumulative_selectivity(selectivity_history_dict):
    cumulative_selectivity = []
    n_bins = 100
    cdf_prob_bins = np.arange(1., n_bins + 1.) / n_bins
    for model_seed in selectivity_history_dict:
        selectivity = selectivity_history_dict[model_seed]['Output']

        nonresponsive_idx = np.where(selectivity==0)[0]
        selectivity[nonresponsive_idx] = 128 #penalize units that are silent across all patterns

        selectivity =1 - selectivity / 128  # convert selectivity to fraction active patterns (per unit)

        selectivity = np.sort(selectivity[:])
        quantiles = [np.quantile(selectivity, pi) for pi in cdf_prob_bins]
        cumulative_selectivity.append(quantiles)

    cumulative_selectivity = np.array(cumulative_selectivity)
    mean_selectivity = np.mean(cumulative_selectivity, axis=0)

    SD = np.std(cumulative_selectivity, axis=0)
    SEM = SD / np.sqrt(cumulative_selectivity.shape[0])

    return mean_selectivity, cdf_prob_bins, SD


def plot_cumulative_sparsity(sparsity_history_dict):
    cumulative_sparsity = []
    n_bins = 100
    cdf_prob_bins = np.arange(1., n_bins + 1.) / n_bins

    for model_seed in sparsity_history_dict:
        sparsity = sparsity_history_dict[model_seed]['Output']

        nonresponsive_idx = np.where(sparsity==0)[0]
        sparsity[nonresponsive_idx[1:]] = 128 # penalize silent patterns (except for the first one, where input is also 0)

        sparsity = 1 - sparsity / 128 # convert sparsity to fraction active units (per pattern)

        sparsity = np.sort(sparsity[:])
        quantiles = [np.quantile(sparsity, pi) for pi in cdf_prob_bins]
        cumulative_sparsity.append(quantiles)

    cumulative_sparsity = np.array(cumulative_sparsity)
    mean_sparsity = np.mean(cumulative_sparsity, axis=0)

    SD = np.std(cumulative_sparsity, axis=0)
    SEM = SD / np.sqrt(cumulative_sparsity.shape[0])

    return  mean_sparsity, cdf_prob_bins, SD


def plot_figure1(num_units_history_dict, sparsity_history_dict, selectivity_history_dict, similarity_matrix_history_dict,
                 weight_history_dict, network_activity_history_dict, color_dict, label_dict, model_seed='1234'):

    mm = 1 / 25.4  # millimeters in inches
    fig1 = plt.figure(figsize=(180 * mm, 200 * mm))
    axes = gs.GridSpec(nrows=4, ncols=6,
                       left=0.08,right=0.94,
                       top = 0.96, bottom = 0.06,
                       wspace=1, hspace=0.6)


    # Input patterns
    ax = fig1.add_subplot(axes[0,0:2])
    num_input_units = num_units_history_dict['Input-Output-lognormal'][model_seed]['Input']
    num_patterns = 2**num_input_units
    sorted_input_patterns = (get_binary_input_patterns(num_input_units, sort=True)).transpose()
    im = ax.imshow(sorted_input_patterns, aspect='auto',cmap='binary')
    ax.set_xticks(np.arange(0, num_patterns+1, num_patterns/4))
    ax.set_yticks(np.arange(0, 7, 1))
    ax.set_xlabel('Pattern ID')
    ax.set_ylabel('Input unit ID')
    ax.set_title('Input Patterns')
    cbar = plt.colorbar(im, ax=ax)

    # Ideal output
    ax = fig1.add_subplot(axes[0,4:6])
    num_output_units = num_units_history_dict['Input-Output-lognormal'][model_seed]['Output']
    im = ax.imshow(np.eye(num_output_units), aspect='auto', cmap='binary')
    ax.set_xticks(np.arange(0, num_output_units+1, num_output_units / 4))
    ax.set_yticks(np.arange(0, num_output_units + 1, num_output_units / 4))
    ax.set_xlabel('Pattern ID')
    ax.set_ylabel('Output unit ID')
    ax.set_title('Ideal network')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Output activity', rotation=270, labelpad=10)


    # Weight distribution
    ax = fig1.add_subplot(axes[1,0:2])
    weight_dict_uniform = weight_history_dict['Input-Output-uniform'][model_seed]['Output']['Input']
    weight_dict_lognormal = weight_history_dict['Input-Output-lognormal'][model_seed]['Output']['Input']
    max_weight_lognormal = np.max(weight_dict_lognormal)
    bin_width = max_weight_lognormal / 80
    hist, edges = np.histogram(weight_dict_uniform,
                               bins=np.arange(-bin_width / 2., max_weight_lognormal + bin_width, bin_width), density=True)
    ax.plot(edges[:-1] + bin_width / 2., hist * bin_width,label=label_dict['Input-Output-uniform'],
                    color=color_dict['Input-Output-uniform'])
    hist, edges = np.histogram(weight_dict_lognormal,
                               bins=np.arange(-bin_width / 2., max_weight_lognormal + bin_width, bin_width), density=True)
    ax.plot(edges[:-1] + bin_width / 2., hist * bin_width,label=label_dict['Input-Output-lognormal'],
                    color=color_dict['Input-Output-lognormal'])
    ax.set_xlabel('Synaptic weights')
    ax.legend(loc='best', frameon=False)


    # Output activity: uniform
    ax = fig1.add_subplot(axes[1, 2:4])
    output_activity_uniform = network_activity_history_dict['Input-Output-uniform'][model_seed]['Output']
    argmax_indices = np.argmax(output_activity_uniform, axis=0) #sort output units according to argmax
    sorted_indices = np.argsort(argmax_indices)
    im = ax.imshow(output_activity_uniform[:, sorted_indices].transpose(), aspect='auto', cmap='binary')
    ax.set_xticks(np.arange(0, num_output_units+1, num_output_units / 4))
    ax.set_yticks(np.arange(0, num_output_units + 1, num_output_units / 4))
    ax.set_xlabel('Pattern ID')
    ax.set_ylabel('Output unit ID')
    ax.set_title('Uniform')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Output activity', rotation=270, labelpad=10)

    # Output activity: log-normal
    ax = fig1.add_subplot(axes[1, 4:6])
    output_activity_lognormal = network_activity_history_dict['Input-Output-lognormal'][model_seed]['Output']
    argmax_indices = np.argmax(output_activity_lognormal, axis=0) #sort output units according to argmax
    sorted_indices = np.argsort(argmax_indices)
    im = ax.imshow(output_activity_lognormal[:, sorted_indices].transpose(), aspect='auto', cmap='binary')
    ax.set_xticks(np.arange(0, num_output_units+1, num_output_units / 4))
    ax.set_yticks(np.arange(0, num_output_units + 1, num_output_units / 4))
    ax.set_xlabel('Pattern ID')
    ax.set_title('Log-normal')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Output activity', rotation=270, labelpad=10)

    # Sparsity
    ax = fig1.add_subplot(axes[2, 0:3])
    scatter_alpha = 0.5
    point_size = 20
    active_unit_count_input = sparsity_history_dict['Input-Output-lognormal'][model_seed]['Input']
    sparsity_input = active_unit_count_input / num_input_units
    ax.scatter((np.arange(0, num_output_units)), sparsity_input,s=point_size, label = label_dict['Input'],
                      color=color_dict['Input'],alpha=scatter_alpha)

    active_unit_count_lognormal = sparsity_history_dict['Input-Output-lognormal'][model_seed]['Output']
    sparsity_lognormal = active_unit_count_lognormal / num_output_units
    ax.scatter((np.arange(0, num_output_units)), sparsity_lognormal,s=point_size, label=label_dict['Input-Output-lognormal'],
                      color=color_dict['Input-Output-lognormal'],alpha=scatter_alpha)

    active_unit_count_uniform = sparsity_history_dict['Input-Output-uniform'][model_seed]['Output']
    sparsity_uniform = active_unit_count_uniform / num_output_units
    ax.scatter((np.arange(0, num_output_units)), sparsity_uniform,s=point_size, label=label_dict['Input-Output-uniform'],
                       color=color_dict['Input-Output-uniform'],alpha=scatter_alpha)
    x = [0,num_output_units]
    y = [1,1]
    ax.plot(x,y,'--',color=color_dict['Ideal'],label=label_dict['Ideal'])
    ax.set_xticks(np.arange(0, num_output_units+1, num_output_units/4))
    ax.set_xlabel('Pattern ID')
    ax.set_ylabel('Fraction active units') #active output neurons count
    ax.set_title('Sparsity')
    ax.legend(loc='best', frameon=False)

    # Selectivity
    ax = fig1.add_subplot(axes[2, 3:6])
    bin_width = 1 / 30

    # num_patterns_selected_input = selectivity_history_dict['Input-Output-lognormal'][model_seed]['Input']
    # selectivity_input = num_patterns_selected_input / 2**num_input_units
    # hist, edges = np.histogram(selectivity_input,
    #                            bins=np.arange(-bin_width / 2., 1 + bin_width, bin_width), density=True)
    # ax.plot(edges[:-1] + bin_width / 2., hist * bin_width,label=label_dict['Input'],
    #                 color=color_dict['Input'])

    num_patterns_selected_lognormal = selectivity_history_dict['Input-Output-lognormal'][model_seed]['Output']
    selectivity_lognormal = num_patterns_selected_lognormal / 2**num_input_units
    hist, edges = np.histogram(selectivity_lognormal,
                               bins=np.arange(-bin_width / 2., 1 + bin_width, bin_width), density=True)
    ax.plot(edges[:-1] + bin_width / 2., hist * bin_width,label=label_dict['Input-Output-lognormal'],
                    color=color_dict['Input-Output-lognormal'])

    num_patterns_selected_uniform = selectivity_history_dict['Input-Output-uniform'][model_seed]['Output']
    selectivity_uniform = num_patterns_selected_uniform / 2**num_input_units
    hist, edges = np.histogram(selectivity_uniform,
                               bins=np.arange(-bin_width / 2., 1 + bin_width, bin_width), density=True)
    ax.plot(edges[:-1] + bin_width / 2., hist * bin_width,label=label_dict['Input-Output-uniform'],
                    color=color_dict['Input-Output-uniform'])
    x = [1/num_patterns, 1/num_patterns]
    y = [0, np.max(hist*bin_width)*1.1] #set ideal line to same height as other distributions
    ax.plot(x,y,'--',linewidth=2,color=color_dict['Ideal'], label=label_dict['Ideal'])
    ax.plot([0.5,0.5],y,linewidth=2,color=color_dict['Input'], label=label_dict['Input'])

    ax.set_xlim([0, 1])
    ax.set_xlabel('Fraction patterns selected')
    ax.set_ylabel('Proportion output units')
    ax.set_title('Selectivity')
    ax.legend(loc='best', frameon=False)

    # Cumulative sparsity
    ax = fig1.add_subplot(axes[3, 0:3])

    description_list = ['Input-Output-uniform','Input-Output-lognormal']
    for description in description_list:
        mean_sparsity, cdf_prob_bins, SD = plot_cumulative_sparsity(sparsity_history_dict[description])
        ax.plot(mean_sparsity, cdf_prob_bins, label=label_dict[description],color=color_dict[description])
        error_min = mean_sparsity - SD
        error_max = mean_sparsity + SD
        ax.fill_betweenx(cdf_prob_bins, error_min, error_max, alpha=0.5,color=color_dict[description])

    sparsity_input = 1 - sparsity_history_dict['Input-Output-lognormal'][model_seed]['Input'] / num_input_units
    sparsity_input = np.sort(sparsity_input[:])
    cumulative_sparsity_input = [np.quantile(sparsity_input, pi) for pi in cdf_prob_bins]
    ax.plot(cumulative_sparsity_input, cdf_prob_bins, label=label_dict['Input'], color=color_dict['Input'])
    ax.set_xlabel('Sparsity')
    ax.set_ylabel('Cumulative probability')

    # Cumulative selectivity
    ax = fig1.add_subplot(axes[3, 3:6])
    for description in description_list:
        mean_selectivity, cdf_prob_bins, SD = plot_cumulative_selectivity(
            selectivity_history_dict[description])
        ax.plot(mean_selectivity, cdf_prob_bins, label=label_dict[description],color=color_dict[description])
        error_min = mean_selectivity - SD
        error_max = mean_selectivity + SD
        ax.fill_betweenx(cdf_prob_bins, error_min, error_max,alpha=0.5,color=color_dict[description])

    selectivity_input = 1 - selectivity_history_dict['Input-Output-lognormal'][model_seed]['Output'] / 2**num_input_units
    selectivity_input = np.sort(selectivity_input[:])
    cumulative_selectivity_input = [np.quantile(selectivity_input, pi) for pi in cdf_prob_bins]
    ax.plot(cumulative_selectivity_input, cdf_prob_bins, label=label_dict['Input'], color=color_dict['Input'])

    ax.set_xlabel('Selectivity')
    ax.set_ylabel('Cumulative probability')
    ax.set_xlim([0,1])

    sns.despine()
    fig1.savefig('figures/Figure1.svg', edgecolor='white', dpi=300, facecolor='white', transparent=True)


def plot_figure2(similarity_matrix_history_dict,num_units_history_dict,color_dict,label_dict,model_seed='1234'):

    mm = 1 / 25.4  # millimeters in inches
    fig2 = plt.figure(figsize=(180 * mm, 110 * mm))
    axes = gs.GridSpec(nrows=2, ncols=6,
                       left=0.08,right=0.94,
                       top = 0.86, bottom = 0.14,
                       wspace=3, hspace=0.4)

    #Similarity matrix
    description_list = ['Input','Input-Output-uniform','Input-Output-lognormal']
    num_patterns = 2**num_units_history_dict['Input-Output-lognormal'][model_seed]['Input']
    for i,description in enumerate(description_list):
        ax = fig2.add_subplot(axes[0, i*2:i*2+2])
        if description=='Input':
            similarity_matrix = similarity_matrix_history_dict['Input-Output-lognormal'][model_seed]['Input']
        else:
            similarity_matrix = similarity_matrix_history_dict[description][model_seed]['Output']
        im = ax.imshow(similarity_matrix, aspect='auto', cmap='viridis', vmin=0, vmax=1)
        ax.set_xticks(np.arange(0, num_patterns+1, num_patterns/4))
        ax.set_yticks(np.arange(0, num_patterns+1, num_patterns/4))
        ax.set_xlabel('Pattern ID')
        ax.set_ylabel('Pattern ID', labelpad=-2)
        ax.set_title(label_dict[description])
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Cosine similarity', rotation=270, labelpad=10)

    #Cosine similarity distribution
    ax = fig2.add_subplot(axes[1, 0:3])
    input_similarity = similarity_matrix_history_dict['Input-Output-lognormal'][model_seed]['Input']
    output_similarity_lognormal = similarity_matrix_history_dict['Input-Output-lognormal'][model_seed]['Output']
    output_similarity_uniform = similarity_matrix_history_dict['Input-Output-uniform'][model_seed]['Output']
    bin_width = 0.04

    invalid_indexes = np.isnan(input_similarity)
    hist, edges = np.histogram(input_similarity[~invalid_indexes],
                               bins=np.arange(-bin_width / 2., 1 + bin_width, bin_width), density=True)
    ax.plot(edges[:-1] + bin_width / 2., hist * bin_width,label='input',
            color=color_dict['Input'])

    invalid_indexes_lognormal = np.isnan(output_similarity_lognormal)
    hist, edges = np.histogram(output_similarity_lognormal[~invalid_indexes_lognormal],
                               bins=np.arange(-bin_width / 2., 1 + bin_width, bin_width), density=True)
    ax.plot(edges[:-1] + bin_width / 2., hist * bin_width,label='log-normal',
                    color=color_dict['Input-Output-lognormal'])

    invalid_indexes_uniform = np.isnan(output_similarity_uniform)
    hist, edges = np.histogram(output_similarity_uniform[~invalid_indexes_uniform],
                               bins=np.arange(-bin_width / 2., 1 + bin_width, bin_width), density=True)
    ax.plot(edges[:-1] + bin_width / 2., hist * bin_width,label='uniform',
                    color=color_dict['Input-Output-uniform'])
    x = [0, 0]
    y = [0, ceil(np.max(hist*bin_width)*10)/10] #set ideal line to same height as other distributions, rounded up
    ax.plot(x, y, color='red', label='Ideal')
    ax.set_xticks(np.arange(0, 1.25, 1 / 4))
    ax.set_xlabel('Cosine similarity')
    ax.set_ylabel('Probability')
    ax.legend(loc='best', frameon=False)

    #Cumulative discriminability
    ax = fig2.add_subplot(axes[1, 3:6])

    description_list = ['Input-Output-uniform','Input-Output-lognormal']
    for description in description_list:
        mean_discriminability, cdf_prob_bins, SD = plot_cumulative_discriminability(similarity_matrix_history_dict[description])
        ax.plot(mean_discriminability, cdf_prob_bins, label=description,color=color_dict[description])
        error_min = mean_discriminability - SD
        error_max = mean_discriminability + SD
        ax.fill_betweenx(cdf_prob_bins, error_min, error_max,alpha=0.4,color=color_dict[description])

    input_similarity_matrix = similarity_matrix_history_dict['Input-Output-uniform'][model_seed]['Input']
    below_diagonal_idx = np.tril_indices_from(input_similarity_matrix, -1)
    input_similarity = input_similarity_matrix[below_diagonal_idx]
    invalid_idx = np.isnan(input_similarity)
    input_similarity = np.sort(input_similarity[~invalid_idx])
    input_discriminability = 1 - input_similarity
    cumulative_input_discriminability = [np.quantile(input_discriminability, pi) for pi in cdf_prob_bins]
    ax.plot(cumulative_input_discriminability, cdf_prob_bins, label=description, color=color_dict['Input'])
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.set_xlabel('Discriminability')
    ax.set_ylabel('Cumulative probability')

    sns.despine()
    fig2.savefig('figures/Figure2.svg', edgecolor='white', dpi=300, facecolor='white', transparent=True)

    return


def plot_figure3(num_units_history_dict,network_activity_history_dict, selectivity_history_dict, similarity_matrix_history_dict,
                 sparsity_history_dict,color_dict,label_dict,model_seed='1234'):

    mm = 1 / 25.4  # millimeters to inches
    fig3 = plt.figure(figsize=(180 * mm, 180 * mm))
    axes = gs.GridSpec(nrows=4, ncols=3,
                       left=0.08,right=0.93,
                       top = 0.94, bottom = 0.1,
                       wspace=0.7, hspace=0.6)

    #Top row:network diagrams for FF, FB, FF+FB

    # Output activity for FF, FB, FF+FB
    description_list = ['FF_Inh','FB_Inh','FF_Inh+FB_Inh']
    num_output_units = num_units_history_dict['FF_Inh'][model_seed]['Output']
    for i,description in enumerate(description_list):
        ax = fig3.add_subplot(axes[1, i])
        output_activity = network_activity_history_dict[description][model_seed]['Output']
        argmax_indices = np.argmax(output_activity, axis=0)
        sorted_indices = np.argsort(argmax_indices)
        im = ax.imshow(output_activity[:, sorted_indices].transpose(), aspect='auto', cmap='binary')
        ax.set_xticks(np.arange(0, num_output_units+1, num_output_units/4))
        ax.set_yticks(np.arange(0, num_output_units+1, num_output_units/4))
        ax.set_xlabel('Pattern ID')
        ax.set_ylabel('Output unit ID')
        ax.set_title(label_dict[description])
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Output activity', rotation=270, labelpad=10)

    #Similarity matrix
    description_list = ['FF_Inh','FB_Inh','FF_Inh+FB_Inh']
    num_patterns = 2**num_units_history_dict['FF_Inh'][model_seed]['Input']
    for i,description in enumerate(description_list):
        ax = fig3.add_subplot(axes[2, i])
        similarity_matrix = similarity_matrix_history_dict[description][model_seed]['Output']
        im = ax.imshow(similarity_matrix, aspect='auto', cmap='viridis',vmin=0, vmax=1)
        ax.set_xticks(np.arange(0, num_patterns+1, num_patterns/4))
        ax.set_yticks(np.arange(0, num_patterns+1, num_patterns/4))
        ax.set_xlabel('Pattern ID')
        ax.set_ylabel('Pattern ID',labelpad=-2)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Cosine similarity', rotation=270, labelpad=10)

    #Cumulative distribution for sparsity (1 - fraction active)
    description_list = ['Input-Output-lognormal','FF_Inh','FB_Inh','FF_Inh+FB_Inh']
    ax = fig3.add_subplot(axes[3, 0])
    for description in description_list:
        mean_sparsity, cdf_prob_bins, SD = plot_cumulative_sparsity(sparsity_history_dict[description])
        ax.plot(mean_sparsity, cdf_prob_bins, label=label_dict[description],color=color_dict[description])
        error_min = mean_sparsity - SD
        error_max = mean_sparsity + SD
        ax.fill_betweenx(cdf_prob_bins, error_min, error_max, alpha=0.2,color=color_dict[description])
    ax.set_xlabel('Sparsity')
    ax.set_ylabel('Cumulative probability')

    #Cumulative distribution for selectivity
    ax = fig3.add_subplot(axes[3, 1])
    for description in description_list:
        mean_selectivity, cdf_prob_bins, SD = plot_cumulative_selectivity(
            selectivity_history_dict[description])
        ax.plot(mean_selectivity, cdf_prob_bins, label=label_dict[description],color=color_dict[description])
        error_min = mean_selectivity - SD
        error_max = mean_selectivity + SD
        ax.fill_betweenx(cdf_prob_bins, error_min, error_max,alpha=0.2,color=color_dict[description])
    ax.set_xlabel('Selectivity')
    ax.set_ylabel('Cumulative probability')

    #Cumulative distribution for discriminability
    ax = fig3.add_subplot(axes[3, 2])
    for description in description_list:
        mean_discriminability, cdf_prob_bins, SD = plot_cumulative_discriminability(similarity_matrix_history_dict[description])
        ax.plot(mean_discriminability, cdf_prob_bins, label=label_dict[description],color=color_dict[description])
        error_min = mean_discriminability - SD
        error_max = mean_discriminability + SD
        ax.fill_betweenx(cdf_prob_bins, error_min, error_max,alpha=0.2,color=color_dict[description])
    ax.set_xlabel('Discriminability')
    ax.set_ylabel('Cumulative probability')
    ax.legend(loc='best',frameon=False)

    sns.despine()
    fig3.savefig('figures/Figure3.svg', edgecolor='white', dpi=300, facecolor='white', transparent=True)


def plot_S1(num_units_history_dict,network_activity_history_dict,color_dict,label_dict,model_seed='1234'):
    'S1, related to Figure 3'

    mm = 1 / 25.4  # millimeters to inches
    figS1 = plt.figure(figsize=(180 * mm, 110 * mm))
    axes = gs.GridSpec(nrows=2, ncols=2,
                       left=0.1,right=0.90,
                       top = 0.94, bottom = 0.1,
                       wspace=0.7, hspace=0.6)

    # Output activity for FF and FB units
    num_output_units = num_units_history_dict['FF_Inh'][model_seed]['Output']

    ax = figS1.add_subplot(axes[0, 0])
    output_activity = network_activity_history_dict['FF_Inh'][model_seed]['FF_Inh']
    argmax_indices = np.argmax(output_activity, axis=0)
    sorted_indices = np.argsort(argmax_indices)
    im = ax.imshow(output_activity[:, sorted_indices].transpose(), aspect='auto', cmap='binary')
    ax.set_xticks(np.arange(0, num_output_units+1, num_output_units/4))
    ax.set_yticks(np.arange(0, 7))
    ax.set_xlabel('Pattern ID')
    ax.set_ylabel('Output unit ID')
    ax.set_title(label_dict['FF_Inh'])
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Output activity', rotation=270, labelpad=10)

    ax = figS1.add_subplot(axes[0, 1])
    output_activity = network_activity_history_dict['FB_Inh'][model_seed]['FB_Inh']
    argmax_indices = np.argmax(output_activity, axis=0)
    sorted_indices = np.argsort(argmax_indices)
    im = ax.imshow(output_activity[:, sorted_indices].transpose(), aspect='auto', cmap='binary')
    ax.set_xticks(np.arange(0, num_output_units+1, num_output_units/4))
    ax.set_yticks(np.arange(0, 7))
    ax.set_xlabel('Pattern ID')
    ax.set_ylabel('Output unit ID')
    ax.set_title(label_dict['FB_Inh'])
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Output activity', rotation=270, labelpad=10)

    population_list = ['FF_Inh','FB_Inh']
    for j, population in enumerate(population_list):
        ax = figS1.add_subplot(axes[1, j])
        output_activity = network_activity_history_dict['FF_Inh+FB_Inh'][model_seed][population]
        argmax_indices = np.argmax(output_activity, axis=0)
        sorted_indices = np.argsort(argmax_indices)
        im = ax.imshow(output_activity.transpose()[sorted_indices, :], aspect='auto', cmap='binary')
        ax.set_xticks(np.arange(0, num_output_units + 1, num_output_units / 4))
        ax.set_yticks(np.arange(0, 7))
        ax.set_xlabel('Pattern ID')
        ax.set_ylabel('Output unit ID')
        ax.set_title(label_dict['FF_Inh+FB_Inh']+': '+population)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Output activity', rotation=270, labelpad=10)

    sns.despine()
    figS1.savefig('figures/S1_F3.svg', edgecolor='white', dpi=300, facecolor='white', transparent=True)


def plot_figure4(num_units_history_dict,network_activity_history_dict, selectivity_history_dict, similarity_matrix_history_dict,
                 sparsity_history_dict,color_dict,label_dict,model_seed='1234'):

    mm = 1 / 25.4  # millimeters to inches
    fig4 = plt.figure(figsize=(180 * mm, 180 * mm))
    axes = gs.GridSpec(nrows=4, ncols=6,
                       left=0.08,right=0.96,
                       top = 0.94, bottom = 0.1,
                       wspace=2, hspace=0.6)

    # Output activity for FF, FB, FF+FB
    description_list = ['FF_Inh+indirect_FB_Inh','FF_Inh+indirect_FB_Inh_c']
    num_output_units = num_units_history_dict['FF_Inh'][model_seed]['Output']
    for i,description in enumerate(description_list):
        ax = fig4.add_subplot(axes[1, i*3:i*3+3])
        output_activity = network_activity_history_dict[description][model_seed]['Output']
        argmax_indices = np.argmax(output_activity, axis=0)
        sorted_indices = np.argsort(argmax_indices)
        im = ax.imshow(output_activity[:, sorted_indices].transpose(), aspect='auto', cmap='binary')
        ax.set_xticks(np.arange(0, num_output_units+1, num_output_units/4))
        ax.set_yticks(np.arange(0, num_output_units+1, num_output_units/4))
        ax.set_xlabel('Pattern ID')
        ax.set_ylabel('Output unit ID')
        ax.set_title(label_dict[description])
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Output activity', rotation=270, labelpad=10)

    #Similarity matrix
    num_patterns = 2**num_units_history_dict['FF_Inh'][model_seed]['Input']
    for i,description in enumerate(description_list):
        ax = fig4.add_subplot(axes[2, i*3:i*3+3])
        similarity_matrix = similarity_matrix_history_dict[description][model_seed]['Output']
        im = ax.imshow(similarity_matrix, aspect='auto', cmap='viridis',vmin=0, vmax=1)
        ax.set_xticks(np.arange(0, num_patterns+1, num_patterns/4))
        ax.set_yticks(np.arange(0, num_patterns+1, num_patterns/4))
        ax.set_xlabel('Pattern ID')
        ax.set_ylabel('Pattern ID',labelpad=-2)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Cosine similarity', rotation=270, labelpad=10)

    #Cumulative distribution for sparsity (1 - fraction active)
    description_list = ['FF_Inh','FF_Inh+indirect_FB_Inh','FF_Inh+indirect_FB_Inh_c']
    ax = fig4.add_subplot(axes[3, 0:2])
    for description in description_list:
        mean_sparsity, cdf_prob_bins, SD = plot_cumulative_sparsity(sparsity_history_dict[description])
        ax.plot(mean_sparsity, cdf_prob_bins, label=label_dict[description],color=color_dict[description])
        error_min = mean_sparsity - SD
        error_max = mean_sparsity + SD
        ax.fill_betweenx(cdf_prob_bins, error_min, error_max, alpha=0.2,color=color_dict[description])
    ax.set_xlabel('Sparsity')
    ax.set_ylabel('Cumulative probability')

    #Cumulative distribution for selectivity
    ax = fig4.add_subplot(axes[3, 2:4])
    for description in description_list:
        mean_selectivity, cdf_prob_bins, SD = plot_cumulative_selectivity(
            selectivity_history_dict[description])
        ax.plot(mean_selectivity, cdf_prob_bins, label=label_dict[description],color=color_dict[description])
        error_min = mean_selectivity - SD
        error_max = mean_selectivity + SD
        ax.fill_betweenx(cdf_prob_bins, error_min, error_max,alpha=0.2,color=color_dict[description])
    ax.set_xlabel('Selectivity')
    ax.set_ylabel('Cumulative probability')

    #Cumulative distribution for discriminability
    ax = fig4.add_subplot(axes[3, 4:6])
    for description in description_list:
        mean_discriminability, cdf_prob_bins, SD = plot_cumulative_discriminability(similarity_matrix_history_dict[description])
        ax.plot(mean_discriminability, cdf_prob_bins, label=label_dict[description],color=color_dict[description])
        error_min = mean_discriminability - SD
        error_max = mean_discriminability + SD
        ax.fill_betweenx(cdf_prob_bins, error_min, error_max,alpha=0.2,color=color_dict[description])
    ax.set_xlabel('Discriminability')
    ax.set_ylabel('Cumulative probability')
    ax.legend(loc='best',frameon=False)

    sns.despine()
    fig4.savefig('figures/Figure4.svg', edgecolor='white', dpi=300, facecolor='white', transparent=True)


def plot_S2(num_units_history_dict,network_activity_history_dict,color_dict,label_dict,model_seed='1234'):
    'S1, related to Figure 3'

    mm = 1 / 25.4  # millimeters to inches
    figS2 = plt.figure(figsize=(180 * mm, 180 * mm))
    axes = gs.GridSpec(nrows=3, ncols=2,
                       left=0.1,right=0.90,
                       top = 0.94, bottom = 0.06,
                       wspace=0.5, hspace=0.6)

    # Output activity for FF_Inh, FB_Inh, and FB_Exc units
    num_patterns = 2**num_units_history_dict['FF_Inh'][model_seed]['Input']

    description_list = ['FF_Inh+indirect_FB_Inh', 'FF_Inh+indirect_FB_Inh_c']
    population_list = ['FF_Inh','FB_Inh','FB_Exc']
    for col,description in enumerate(description_list):
        for row,population in enumerate(population_list):
            ax = figS2.add_subplot(axes[row, col])
            output_activity = network_activity_history_dict[description][model_seed][population]
            argmax_indices = np.argmax(output_activity, axis=0)
            sorted_indices = np.argsort(argmax_indices)
            im = ax.imshow(output_activity[:, sorted_indices].transpose(), aspect='auto', cmap='binary')
            num_units = num_units_history_dict[description][model_seed][population]
            ax.set_xticks(np.arange(0, num_patterns+1, num_patterns/4))
            ax.set_yticks(np.arange(0, num_units))
            ax.set_xlabel('Pattern ID')
            ax.set_ylabel('Output unit ID')
            ax.set_title(label_dict[description]+': \n'+population)
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Output activity', rotation=270, labelpad=10)

    sns.despine()
    figS2.savefig('figures/S2_F4.svg', edgecolor='white', dpi=300, facecolor='white', transparent=True)


def plot_figure5(num_units_history_dict,network_activity_history_dict, selectivity_history_dict, similarity_matrix_history_dict,
                 sparsity_history_dict,color_dict,label_dict,model_seed='1234'):

    mm = 1 / 25.4  # millimeters to inches
    fig5 = plt.figure(figsize=(180 * mm, 180 * mm))
    axes = gs.GridSpec(nrows=4, ncols=6,
                       left=0.08,right=0.96,
                       top = 0.94, bottom = 0.1,
                       wspace=2, hspace=0.6)

    # Output activity for FF, FB, FF+FB
    description_list = ['FF_Inh+indirect_FB_Inh+FB_Exc','FF_Inh+indirect_FB_Inh+FB_Exc_b']
    num_output_units = num_units_history_dict['FF_Inh'][model_seed]['Output']
    for i,description in enumerate(description_list):
        ax = fig5.add_subplot(axes[1, i*3:i*3+3])
        output_activity = network_activity_history_dict[description][model_seed]['Output']
        argmax_indices = np.argmax(output_activity, axis=0)
        sorted_indices = np.argsort(argmax_indices)
        im = ax.imshow(output_activity[:, sorted_indices].transpose(), aspect='auto', cmap='binary')
        ax.set_xticks(np.arange(0, num_output_units+1, num_output_units/4))
        ax.set_yticks(np.arange(0, num_output_units+1, num_output_units/4))
        ax.set_xlabel('Pattern ID')
        ax.set_ylabel('Output unit ID')
        ax.set_title(label_dict[description])
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Output activity', rotation=270, labelpad=10)

    #Similarity matrix
    num_patterns = 2**num_units_history_dict['FF_Inh'][model_seed]['Input']
    for i,description in enumerate(description_list):
        ax = fig5.add_subplot(axes[2, i*3:i*3+3])
        similarity_matrix = similarity_matrix_history_dict[description][model_seed]['Output']
        im = ax.imshow(similarity_matrix, aspect='auto', cmap='viridis',vmin=0, vmax=1)
        ax.set_xticks(np.arange(0, num_patterns+1, num_patterns/4))
        ax.set_yticks(np.arange(0, num_patterns+1, num_patterns/4))
        ax.set_xlabel('Pattern ID')
        ax.set_ylabel('Pattern ID',labelpad=-2)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Cosine similarity', rotation=270, labelpad=10)

    #Cumulative distribution for sparsity (1 - fraction active)
    description_list = ['FF_Inh+indirect_FB_Inh','FF_Inh+indirect_FB_Inh+FB_Exc','FF_Inh+indirect_FB_Inh+FB_Exc_b']
    ax = fig5.add_subplot(axes[3, 0:2])
    for description in description_list:
        mean_sparsity, cdf_prob_bins, SD = plot_cumulative_sparsity(sparsity_history_dict[description])
        ax.plot(mean_sparsity, cdf_prob_bins, label=label_dict[description],color=color_dict[description])
        error_min = mean_sparsity - SD
        error_max = mean_sparsity + SD
        ax.fill_betweenx(cdf_prob_bins, error_min, error_max, alpha=0.2,color=color_dict[description])
    ax.set_xlabel('Sparsity')
    ax.set_ylabel('Cumulative probability')

    #Cumulative distribution for selectivity
    ax = fig5.add_subplot(axes[3, 2:4])
    for description in description_list:
        mean_selectivity, cdf_prob_bins, SD = plot_cumulative_selectivity(
            selectivity_history_dict[description])
        ax.plot(mean_selectivity, cdf_prob_bins, label=label_dict[description],color=color_dict[description])
        error_min = mean_selectivity - SD
        error_max = mean_selectivity + SD
        ax.fill_betweenx(cdf_prob_bins, error_min, error_max,alpha=0.2,color=color_dict[description])
    ax.set_xlabel('Selectivity')
    ax.set_ylabel('Cumulative probability')

    #Cumulative distribution for discriminability
    ax = fig5.add_subplot(axes[3, 4:6])
    for description in description_list:
        mean_discriminability, cdf_prob_bins, SD = plot_cumulative_discriminability(similarity_matrix_history_dict[description])
        ax.plot(mean_discriminability, cdf_prob_bins, label=label_dict[description],color=color_dict[description])
        error_min = mean_discriminability - SD
        error_max = mean_discriminability + SD
        ax.fill_betweenx(cdf_prob_bins, error_min, error_max,alpha=0.2,color=color_dict[description])
    ax.set_xlabel('Discriminability')
    ax.set_ylabel('Cumulative probability')
    ax.legend(loc='best',frameon=False)

    sns.despine()
    fig5.savefig('figures/Figure5.svg', edgecolor='white', dpi=300, facecolor='white', transparent=True)


def plot_S3(num_units_history_dict,network_activity_history_dict,color_dict,label_dict,model_seed='1234'):
    'S3, related to Figure 5'

    mm = 1 / 25.4  # millimeters to inches
    figS3 = plt.figure(figsize=(180 * mm, 180 * mm))
    axes = gs.GridSpec(nrows=3, ncols=2,
                       left=0.1,right=0.90,
                       top = 0.92, bottom = 0.06,
                       wspace=0.5, hspace=0.8)

    # Output activity for FF_Inh, FB_Inh, and FB_Exc units
    num_patterns = 2**num_units_history_dict['FF_Inh'][model_seed]['Input']

    description_list = ['FF_Inh+indirect_FB_Inh+FB_Exc','FF_Inh+indirect_FB_Inh+FB_Exc_b']
    population_list = ['FF_Inh','FB_Inh','FB_Exc']
    for col,description in enumerate(description_list):
        for row,population in enumerate(population_list):
            ax = figS3.add_subplot(axes[row, col])
            output_activity = network_activity_history_dict[description][model_seed][population]
            argmax_indices = np.argmax(output_activity, axis=0)
            sorted_indices = np.argsort(argmax_indices)
            im = ax.imshow(output_activity[:, sorted_indices].transpose(), aspect='auto', cmap='binary')
            num_units = num_units_history_dict[description][model_seed][population]
            ax.set_xticks(np.arange(0, num_patterns+1, num_patterns/4))
            ax.set_yticks(np.arange(0, num_units))
            ax.set_xlabel('Pattern ID')
            ax.set_ylabel('Output unit ID')
            ax.set_title(label_dict[description]+': \n'+population)
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Output activity', rotation=270, labelpad=10)

    sns.despine()
    figS3.savefig('figures/S3_F5.svg', edgecolor='white', dpi=300, facecolor='white', transparent=True)


def plotS4(network_activity_dynamics_history_dict,color_dict,label_dict,model_seed='1234'):
    'S4, related to Figure 5: population dynamics'
    mm = 1 / 25.4  # millimeters to inches
    figS4 = plt.figure(figsize=(180 * mm, 120 * mm))
    axes = gs.GridSpec(nrows=2, ncols=2,
                       left=0.1,right=0.90,
                       top = 0.92, bottom = 0.1,
                       wspace=0.5, hspace=0.6)

    population_list = ['Output','FB_Exc','FF_Inh','FB_Inh']
    description = 'FF_Inh+indirect_FB_Inh+FB_Exc_b'

    for i,population in enumerate(population_list):
        mean_across_patterns = np.mean(network_activity_dynamics_history_dict[description][model_seed][population],axis=0)
        mean_activity = np.mean(mean_across_patterns,axis=0)

        col = i%2
        if i<2:
            row = 0
        else:
            row = 1
        ax = figS4.add_subplot(axes[row, col])
        ax.plot(mean_activity)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Mean activity')
        ax.set_title(population)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    figS4.savefig('figures/S4_F5.svg', edgecolor='white', dpi=300, facecolor='white', transparent=True)


#############################################################################

@click.command()
@click.option("--data_file_path", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--dynamics_file_path", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--model_seed", type=str, default='1234')

def main(data_file_path, dynamics_file_path, model_seed):

    # Import data
    _,num_units_history_dict,_,_,weight_history_dict, network_activity_history_dict, sparsity_history_dict, \
        similarity_matrix_history_dict, selectivity_history_dict,fraction_active_patterns_history_dict,\
        _ = import_slice_data(data_file_path)
    network_activity_dynamics_history_dict =  import_dynamic_activity(dynamics_file_path)

    # Specify figure parameters (colors & labels)
    colorbrewer_list = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c',
                        '#fdbf6f','#ff7f00','#cab2d6']
    color_dict = {'Ideal':'red',
                  'Input': 'magenta',
                  'Input-Output-uniform': colorbrewer_list[0],
                  'Input-Output-lognormal': colorbrewer_list[1],
                  'FF_Inh': colorbrewer_list[2],
                  'FB_Inh': colorbrewer_list[3],
                  'FF_Inh+FB_Inh': colorbrewer_list[4],
                  'FF_Inh+indirect_FB_Inh': colorbrewer_list[5],
                  'FF_Inh+indirect_FB_Inh_c': colorbrewer_list[6],
                  'FF_Inh+indirect_FB_Inh+FB_Exc': colorbrewer_list[7],
                  'FF_Inh+indirect_FB_Inh+FB_Exc_b': colorbrewer_list[8]}
    label_dict = {'Ideal':'Ideal',
                  'Input': 'Input',
                  'Input-Output-uniform': 'No inhibition \n (uniform exc)',
                  'Input-Output-lognormal': 'No inhibition \n (log-normal exc)',
                  'FF_Inh': 'Feedforward inhibition',
                  'FB_Inh': 'Feedback inhibition',
                  'FF_Inh+FB_Inh': 'FF+FB Inhibition',
                  'FF_Inh+indirect_FB_Inh': 'Inhibition: FF + indirect FB',
                  'FF_Inh+indirect_FB_Inh_b': 'FF_Inh+indirect_FB_Inh_b',
                  'FF_Inh+indirect_FB_Inh_c': 'No recurrent connection',
                  'FF_Inh+indirect_FB_Inh+FB_Exc': 'Mossy cells: FB excitation',
                  'FF_Inh+indirect_FB_Inh+FB_Exc_b': 'Mossy cells: \n FB excitation + inhibition'}

    # # Generate figures
    # plotS4(network_activity_dynamics_history_dict,color_dict,label_dict)
    #
    # plot_S3(num_units_history_dict,network_activity_history_dict,color_dict,label_dict,model_seed)
    #
    # plot_figure5(num_units_history_dict, network_activity_history_dict, selectivity_history_dict, similarity_matrix_history_dict,
    #              sparsity_history_dict, color_dict, label_dict, model_seed='1234')
    #
    # plot_S2(num_units_history_dict,network_activity_history_dict,color_dict,label_dict,model_seed)
    #
    # plot_figure4(num_units_history_dict,network_activity_history_dict, selectivity_history_dict, similarity_matrix_history_dict,
    #              sparsity_history_dict,color_dict,label_dict,model_seed)
    #
    # plot_S1(num_units_history_dict,network_activity_history_dict,color_dict,label_dict,model_seed)
    #
    # plot_figure3(num_units_history_dict,network_activity_history_dict, selectivity_history_dict, similarity_matrix_history_dict,
    #              sparsity_history_dict,color_dict,label_dict,model_seed)
    #
    # plot_figure2(similarity_matrix_history_dict,num_units_history_dict,color_dict,label_dict,model_seed)

    plot_figure1(num_units_history_dict, sparsity_history_dict, selectivity_history_dict, similarity_matrix_history_dict,
                 weight_history_dict, network_activity_history_dict,color_dict,label_dict,model_seed)

    plt.show()

if __name__ == '__main__':
    # this extra flag stops the click module from forcing system exit when python is in interactive mode
    main(standalone_mode=False)
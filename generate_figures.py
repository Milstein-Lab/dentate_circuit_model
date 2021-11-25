
# python generate_figures.py --data_file_path=data/20211116_exported_dentate_model_data.hdf5 --dynamics_file_path=data/20211116_exported_dentate_model_data_dynamics.hdf5 --plot

import click
import numpy as np
from scipy import stats
from math import ceil
import h5py
from copy import deepcopy
from optimize_dynamic_model import analyze_slice, get_binary_input_patterns
import os

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

    return cumulative_discriminability.flatten(), mean_discriminability, cdf_prob_bins, SD


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

    return cumulative_selectivity.flatten(), mean_selectivity, cdf_prob_bins, SD


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

    return  cumulative_sparsity.flatten(), mean_sparsity, cdf_prob_bins, SD


def plot_figure1(num_units_history_dict, sparsity_history_dict, selectivity_history_dict, similarity_matrix_history_dict,
                 weight_history_dict, network_activity_history_dict, color_dict, label_dict, model_seed='1234'):

    mm = 1 / 25.4  # millimeters in inches
    fig = plt.figure(figsize=(180 * mm, 120 * mm))
    axes = gs.GridSpec(nrows=3, ncols=5,
                       left=0.05,right=0.98,
                       top = 0.95, bottom = 0.2,
                       wspace=0.8, hspace=1.2)

    fontsize = 8
    num_input_units = num_units_history_dict['Input-Output-lognormal'][model_seed]['Input']
    num_patterns = 2**num_input_units
    num_output_units = num_units_history_dict['Input-Output-lognormal'][model_seed]['Output']

    # Input patterns
    ax = fig.add_subplot(axes[1,0])
    sorted_input_patterns = (get_binary_input_patterns(num_input_units, sort=True)).transpose()
    im = ax.imshow(sorted_input_patterns, aspect='auto',cmap='binary',interpolation="nearest")
    ax.set_xticks([0, num_patterns])
    ax.set_yticks([-0.5, num_input_units - 0.5])
    ax.set_yticklabels([0, num_input_units])
    ax.tick_params(labelsize=fontsize)
    ax.set_xlabel('Pattern ID',fontsize=fontsize,labelpad=0)
    ax.set_ylabel('Input unit ID',fontsize=fontsize)
    ax.set_title('Input activity',fontsize=fontsize)

    # Ideal output
    ax = fig.add_subplot(axes[0,1])
    im = ax.imshow(np.eye(num_output_units), aspect='equal', cmap='binary',interpolation="nearest")
    ax.set_xticks([0, num_output_units])
    ax.set_yticks([0, num_patterns])
    ax.tick_params(labelsize=fontsize)
    ax.set_xlabel('Pattern ID',fontsize=fontsize,labelpad=0)
    ax.set_ylabel('Output unit ID',fontsize=fontsize, labelpad=-1)
    ax.set_title('Ideal output',fontsize=fontsize)

    # Weight distribution
    ax = fig.add_subplot(axes[1,1])
    weight_dict_uniform = weight_history_dict['Input-Output-uniform'][model_seed]['Output']['Input']
    weight_dict_lognormal = weight_history_dict['Input-Output-lognormal'][model_seed]['Output']['Input']
    max_weight_lognormal = np.max(weight_dict_lognormal)
    bin_width = max_weight_lognormal / 80
    hist, edges = np.histogram(weight_dict_uniform,
                               bins=np.arange(-bin_width / 2., max_weight_lognormal + bin_width, bin_width), density=True)
    ax.plot(edges[:-1] + bin_width / 2., hist * bin_width,label='Uniform \nweights',
                    color=color_dict['Input-Output-uniform'])
    hist, edges = np.histogram(weight_dict_lognormal,
                               bins=np.arange(-bin_width / 2., max_weight_lognormal + bin_width, bin_width), density=True)
    ax.plot(edges[:-1] + bin_width / 2., hist * bin_width,label='Log-normal \nweights',
                    color=color_dict['Input-Output-lognormal'])
    ax.set_xlabel('Synaptic weights',fontsize=fontsize,labelpad=0)
    ax.set_ylabel('Proportion of \nweights',fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    ax.legend(loc='best', frameon=False,fontsize=fontsize,handlelength=1)

    # Output activity: uniform
    ax = fig.add_subplot(axes[0, 2])
    output_activity_uniform = network_activity_history_dict['Input-Output-uniform'][model_seed]['Output']
    argmax_indices = np.argmax(output_activity_uniform, axis=0) #sort output units according to argmax
    sorted_indices = np.argsort(argmax_indices)
    im = ax.imshow(output_activity_uniform[:, sorted_indices].transpose(), aspect='equal', cmap='binary',vmax=0.5,vmin=0,interpolation="nearest")
    ax.set_xticks([0, num_patterns])
    ax.set_yticks([0, num_patterns])
    ax.tick_params(labelsize=fontsize)
    ax.set_xlabel('Pattern ID',fontsize=fontsize,labelpad=0)
    ax.set_ylabel('Output unit ID',fontsize=fontsize,labelpad=-1)
    ax.set_title(label_dict['Input-Output-uniform'],fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    cbar = plt.colorbar(im, ax=ax,ticks=[0, 0.5])
    cbar.ax.set_yticklabels([0,0.5])
    cbar.set_label('Output activity', rotation=270, labelpad=7, fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)

    # Output activity: log-normal
    ax = fig.add_subplot(axes[1, 2])
    output_activity_lognormal = network_activity_history_dict['Input-Output-lognormal'][model_seed]['Output']
    argmax_indices = np.argmax(output_activity_lognormal, axis=0) #sort output units according to argmax
    sorted_indices = np.argsort(argmax_indices)
    im = ax.imshow(output_activity_lognormal[:, sorted_indices].transpose(), aspect='equal', cmap='binary',vmax=0.5,vmin=0,interpolation="nearest")
    ax.set_xticks([0, num_patterns])
    ax.set_yticks([0, num_patterns])
    ax.tick_params(labelsize=fontsize)
    ax.set_xlabel('Pattern ID',fontsize=fontsize,labelpad=0)
    ax.set_ylabel('Output unit ID',fontsize=fontsize,labelpad=-1)
    ax.set_title(label_dict['Input-Output-lognormal'],fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    cbar = plt.colorbar(im, ax=ax,ticks=[0, 0.5])
    cbar.ax.set_yticklabels([0,0.5])
    cbar.set_label('Output activity', rotation=270, labelpad=7, fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)

    # Sparsity scatter
    ax = fig.add_subplot(axes[0, 3])
    scatter_alpha = 0.5
    point_size = 5
    active_unit_count_input = sparsity_history_dict['Input-Output-lognormal'][model_seed]['Input']
    sparsity_input = active_unit_count_input / num_input_units
    ax.scatter(np.arange(0, num_output_units), sparsity_input,s=point_size, label = label_dict['Input'],
                      color=color_dict['Input'],alpha=scatter_alpha)

    active_unit_count_lognormal = sparsity_history_dict['Input-Output-lognormal'][model_seed]['Output']
    sparsity_lognormal = active_unit_count_lognormal / num_output_units
    ax.scatter(np.arange(0, num_output_units), sparsity_lognormal,s=point_size, label=label_dict['Input-Output-lognormal'],
                      color=color_dict['Input-Output-lognormal'],alpha=scatter_alpha)

    active_unit_count_uniform = sparsity_history_dict['Input-Output-uniform'][model_seed]['Output']
    sparsity_uniform = active_unit_count_uniform / num_output_units
    ax.scatter(np.arange(0, num_output_units), sparsity_uniform,s=point_size, label=label_dict['Input-Output-uniform'],
                       color=color_dict['Input-Output-uniform'],alpha=scatter_alpha)
    x = [0, num_patterns]
    y = [1/num_output_units, 1/num_output_units]
    ax.plot(x,y,'--',color=color_dict['Ideal'],label=label_dict['Ideal'])
    ax.set_ylim([0, 1])

    ax.set_xticks([0,num_patterns])
    ax.set_xlim([0, num_output_units])
    ax.set_yticks([0, 1])
    ax.tick_params(labelsize=fontsize)
    ax.set_xlabel('Pattern ID',fontsize=fontsize,labelpad=0)
    ax.set_ylabel('Fraction \nactive units',fontsize=fontsize,labelpad=0) #active output neurons count

    # Cumulative sparsity
    ax = fig.add_subplot(axes[1, 3])

    description_list = ['Input-Output-uniform','Input-Output-lognormal']
    cumulative_sparsity_dict = {}
    for description in description_list:
        cumulative_sparsity, mean_sparsity, cdf_prob_bins, SD = plot_cumulative_sparsity(sparsity_history_dict[description])
        cumulative_sparsity_dict[description] = cumulative_sparsity
        ax.plot(mean_sparsity, cdf_prob_bins, label=label_dict[description],color=color_dict[description])
        error_min = mean_sparsity - SD
        error_max = mean_sparsity + SD
        ax.fill_betweenx(cdf_prob_bins, error_min, error_max, alpha=0.5,color=color_dict[description])

    sparsity_input = 1 - sparsity_history_dict['Input-Output-lognormal'][model_seed]['Input'] / num_input_units
    sparsity_input = np.sort(sparsity_input[:])
    cumulative_sparsity_input = [np.quantile(sparsity_input, pi) for pi in cdf_prob_bins]
    cumulative_sparsity_dict['Input'] = cumulative_sparsity_input

    s, p1 = stats.ks_2samp(cumulative_sparsity_dict['Input-Output-uniform'],
                          cumulative_sparsity_dict['Input-Output-lognormal'])
    s, p2 = stats.ks_2samp(cumulative_sparsity_dict['Input'],
                          cumulative_sparsity_dict['Input-Output-uniform'])
    s, p3 = stats.ks_2samp(cumulative_sparsity_dict['Input'],
                          cumulative_sparsity_dict['Input-Output-lognormal'])
    path_to_file = 'ks_tests.txt'
    mode = 'a' if os.path.exists(path_to_file) else 'w'
    with open(path_to_file, mode) as f:
        f.write(f"\nFig1 Sparsity Stats:"
                f"\nUniform Vs Lognormal: p = {p1}"
                f"\nInput Vs Uniform: p = {p2}"
                f"\nInput Vs Lognormal: p = {p3}")

    ax.plot(cumulative_sparsity_input, cdf_prob_bins, label=label_dict['Input'], color=color_dict['Input'])
    ax.plot([1,1],[0,1],'--',color=color_dict['Ideal'],label=label_dict['Ideal'])

    ax.set_xlabel('Sparsity',fontsize=fontsize,labelpad=0)
    ax.set_ylabel('Cumulative \nprobability',fontsize=fontsize,labelpad=0)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.tick_params(labelsize=fontsize)

    # Selectivity scatter
    ax = fig.add_subplot(axes[0, 4])
    x = [0, num_output_units]
    y = [0.5, 0.5]
    ax.plot(x,y,'--',color=color_dict['Input'],label=label_dict['Input'])
    x = [0, num_output_units]
    y = [1/num_patterns, 1/num_patterns]
    ax.plot(x,y,'--',color=color_dict['Ideal'],label=label_dict['Ideal'])

    active_pattern_count = selectivity_history_dict['Input-Output-uniform'][model_seed]['Output']
    fraction_active = active_pattern_count / num_patterns
    fraction_active = np.sort(fraction_active)
    ax.scatter(np.arange(0, num_patterns), fraction_active, s=point_size, label=label_dict['Input-Output-uniform'],
               color=color_dict['Input-Output-uniform'], alpha=scatter_alpha)

    active_pattern_count = selectivity_history_dict['Input-Output-lognormal'][model_seed]['Output']
    fraction_active = active_pattern_count / num_patterns
    fraction_active = np.sort(fraction_active)
    ax.scatter(np.arange(0, num_patterns), fraction_active,s=point_size, label=label_dict['Input-Output-lognormal'],
                      color=color_dict['Input-Output-lognormal'],alpha=scatter_alpha)

    ax.set_ylim([0, 1])
    ax.set_xlim([0, num_output_units])
    ax.set_xticks([0, num_output_units])
    ax.set_yticks([0, 1])
    ax.tick_params(labelsize=fontsize)
    ax.set_xlabel('Output unit ID (sorted)',fontsize=fontsize,labelpad=0)
    ax.set_ylabel('Fraction \nactive patterns',fontsize=fontsize,labelpad=0) #active output neurons count
    ax.legend(loc='best', frameon=False,fontsize=fontsize,handlelength=1)

    # Cumulative selectivity
    ax = fig.add_subplot(axes[1, 4])
    cumulative_selectivity_dict = {}
    for description in description_list:
        cumulative_selectivity, mean_selectivity, cdf_prob_bins, SD = plot_cumulative_selectivity(selectivity_history_dict[description])
        cumulative_selectivity_dict[description] = cumulative_selectivity
        ax.plot(mean_selectivity, cdf_prob_bins, label=label_dict[description],color=color_dict[description])
        error_min = mean_selectivity - SD
        error_max = mean_selectivity + SD
        ax.fill_betweenx(cdf_prob_bins, error_min, error_max,alpha=0.5,color=color_dict[description])

    selectivity_input = 1 - selectivity_history_dict['Input-Output-lognormal'][model_seed]['Input'] / 2**num_input_units
    selectivity_input = np.sort(selectivity_input[:])
    cumulative_selectivity_input = [np.quantile(selectivity_input, pi) for pi in cdf_prob_bins]
    cumulative_selectivity_dict['Input'] = cumulative_selectivity_input

    s, p1 = stats.ks_2samp(cumulative_selectivity_dict['Input-Output-uniform'],
                          cumulative_selectivity_dict['Input-Output-lognormal'])
    s, p2 = stats.ks_2samp(cumulative_selectivity_dict['Input'],
                          cumulative_selectivity_dict['Input-Output-uniform'])
    s, p3 = stats.ks_2samp(cumulative_selectivity_dict['Input'],
                          cumulative_selectivity_dict['Input-Output-lognormal'])
    with open(path_to_file, 'a') as f:
        f.write("\nFig1 Selectivity Stats:"
                f"\nUniform Vs Lognormal: p = {p1}"
                f"\nInput Vs Uniform: p = {p2}"
                f"\nInput Vs Lognormal: p = {p3}\n")

    ax.plot(cumulative_selectivity_input, cdf_prob_bins, label=label_dict['Input'], color=color_dict['Input'])
    ax.plot([1,1],[0,1],'--',color=color_dict['Ideal'],label=label_dict['Ideal'])

    ax.set_xlabel('Selectivity',fontsize=fontsize,labelpad=0)
    ax.set_ylabel('Cumulative \nprobability',fontsize=fontsize,labelpad=0)
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.set_xticks([0,1])
    ax.set_yticks([0,1])
    ax.tick_params(labelsize=fontsize)

    sns.despine()
    fig.savefig('figures/Figure1.svg', edgecolor='white', dpi=300, facecolor='white', transparent=True)
    # fig.savefig('figures/Figure1.png', edgecolor='white', dpi=300, facecolor='white', transparent=True)


def plot_figure2(similarity_matrix_history_dict,num_units_history_dict,color_dict,label_dict,model_seed='1234'):

    mm = 1 / 25.4  # millimeters in inches
    fig = plt.figure(figsize=(180 * mm, 120 * mm))
    axes = gs.GridSpec(nrows=3, ncols=5,
                       left=0.05,right=0.98,
                       top = 0.95, bottom = 0.2,
                       wspace=0.8, hspace=1.2)

    fontsize = 8

    label_dict['Input-Output-uniform'] = 'Uniform weights'
    label_dict['Input-Output-lognormal'] = 'Log-normal weights'


    #Similarity matrix
    description_list = ['Input','Input-Output-uniform','Input-Output-lognormal']
    num_patterns = 2**num_units_history_dict['Input-Output-lognormal'][model_seed]['Input']
    for i,description in enumerate(description_list):
        ax = fig.add_subplot(axes[0, i])
        if description=='Input':
            similarity_matrix = similarity_matrix_history_dict['Input-Output-lognormal'][model_seed]['Input']
        else:
            similarity_matrix = similarity_matrix_history_dict[description][model_seed]['Output']
        im = ax.imshow(similarity_matrix, aspect='equal', cmap='viridis', vmin=0, vmax=1,interpolation="nearest")
        ax.set_xticks([0, num_patterns])
        ax.set_yticks([0, num_patterns])
        ax.set_xlabel('Pattern ID',fontsize=fontsize,labelpad=0)
        ax.set_ylabel('Pattern ID',fontsize=fontsize , labelpad=-2)
        ax.set_title(label_dict[description],fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        if i==2:
            cbar = plt.colorbar(im, ax=ax, ticks=[0, 1])
            cbar.ax.set_yticklabels([0, 1])
            cbar.set_label('Cosine similarity', rotation=270, labelpad=8,fontsize=fontsize)
            cbar.ax.tick_params(labelsize=fontsize)

    #Cosine similarity distribution
    ax = fig.add_subplot(axes[0,3])
    input_similarity = similarity_matrix_history_dict['Input-Output-lognormal'][model_seed]['Input']
    output_similarity_lognormal = similarity_matrix_history_dict['Input-Output-lognormal'][model_seed]['Output']
    output_similarity_uniform = similarity_matrix_history_dict['Input-Output-uniform'][model_seed]['Output']
    bin_width = 0.04

    invalid_indexes = np.isnan(input_similarity)
    hist, edges = np.histogram(input_similarity[~invalid_indexes],
                               bins=np.arange(-bin_width / 2., 1 + bin_width, bin_width), density=True)
    ax.plot(edges[:-1] + bin_width / 2., hist * bin_width,label=label_dict['Input'],
            color=color_dict['Input'])

    x = [0, 0]
    y = [0, np.max(hist*bin_width)] #set ideal line to same height as other distributions, rounded up
    ax.plot(x, y, '--', color=color_dict['Ideal'], label=label_dict['Ideal'])

    invalid_indexes_uniform = np.isnan(output_similarity_uniform)
    hist, edges = np.histogram(output_similarity_uniform[~invalid_indexes_uniform],
                               bins=np.arange(-bin_width / 2., 1 + bin_width, bin_width), density=True)
    ax.plot(edges[:-1] + bin_width / 2., hist * bin_width,label=label_dict['Input-Output-uniform'],
                    color=color_dict['Input-Output-uniform'])

    invalid_indexes_lognormal = np.isnan(output_similarity_lognormal)
    hist, edges = np.histogram(output_similarity_lognormal[~invalid_indexes_lognormal],
                               bins=np.arange(-bin_width / 2., 1 + bin_width, bin_width), density=True)
    ax.plot(edges[:-1] + bin_width / 2., hist * bin_width,label=label_dict['Input-Output-lognormal'],
                    color=color_dict['Input-Output-lognormal'])

    ax.set_xticks([0,1])
    ax.set_xlim([0,1])
    ax.tick_params(labelsize=fontsize)
    ax.set_xlabel('Cosine similarity', fontsize=fontsize,labelpad=0)
    ax.set_ylabel('Probability', fontsize=fontsize,labelpad=0)
    ax.legend(loc='best', frameon=False, fontsize=fontsize,handlelength=1)

    #Cumulative discriminability
    ax = fig.add_subplot(axes[0,4])

    description_list = ['Input-Output-uniform','Input-Output-lognormal']
    cumulative_discriminability_dict = {}
    for description in description_list:
        cumulative_discriminability, mean_discriminability, cdf_prob_bins, SD = plot_cumulative_discriminability(similarity_matrix_history_dict[description])
        cumulative_discriminability_dict[description] = cumulative_discriminability
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
    cumulative_discriminability_dict['Input'] = cumulative_input_discriminability

    s, p1 = stats.ks_2samp(cumulative_discriminability_dict['Input-Output-uniform'],
                          cumulative_discriminability_dict['Input-Output-lognormal'])
    s, p2 = stats.ks_2samp(cumulative_discriminability_dict['Input'],
                          cumulative_discriminability_dict['Input-Output-uniform'])
    s, p3 = stats.ks_2samp(cumulative_discriminability_dict['Input'],
                          cumulative_discriminability_dict['Input-Output-lognormal'])
    path_to_file = 'ks_tests.txt'
    mode = 'a' if os.path.exists(path_to_file) else 'w'
    with open(path_to_file, mode) as f:
        f.write(f"\nFig2 Discriminability Stats:"
                f"\nUniform Vs Lognormal: p = {p1}"
                f"\nInput Vs Uniform: p = {p2}"
                f"\nInput Vs Lognormal: p = {p3}\n")

    ax.plot(cumulative_input_discriminability, cdf_prob_bins, label=description, color=color_dict['Input'])
    ax.plot([1,1],[0,1],'--', color=color_dict['Ideal'], label=label_dict['Ideal'])

    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.set_xticks([0,1])
    ax.set_yticks([0,1])
    ax.tick_params(labelsize=fontsize)
    ax.set_xlabel('Discriminability',fontsize=fontsize,labelpad=0)
    ax.set_ylabel('Cumulative \nprobability',fontsize=fontsize,labelpad=0)

    sns.despine()
    fig.savefig('figures/Figure2.svg', edgecolor='white', dpi=300, facecolor='white', transparent=True)
    # fig.savefig('figures/Figure2.png', edgecolor='white', dpi=300, facecolor='white', transparent=True)


def plot_figure3(num_units_history_dict, weight_history_dict, network_activity_history_dict, selectivity_history_dict, similarity_matrix_history_dict,
                 sparsity_history_dict,color_dict,label_dict,model_seed='1234'):

    mm = 1 / 25.4  # millimeters to inches
    fig = plt.figure(figsize=(180 * mm, 120 * mm))
    axes = gs.GridSpec(nrows=3, ncols=5,
                       left=0.05,right=0.98,
                       top = 0.95, bottom = 0.2,
                       wspace=0.8, hspace=1.2)


    fontsize = 8
    description_list = ['FF_Inh','FF_Inh_no_sel_loss']

    # Model diagrams & titles
    for i,description in enumerate(description_list):
        ax = fig.add_subplot(axes[i, 0])
        ax.axis('off')
        ax.set_title(label_dict[description], fontsize=fontsize)

    # Output activity for FF, FB, FF+FB
    num_output_units = num_units_history_dict['FF_Inh'][model_seed]['Output']
    for i,description in enumerate(description_list):
        ax = fig.add_subplot(axes[i, 1])
        output_activity = network_activity_history_dict[description][model_seed]['Output']
        argmax_indices = np.argmax(output_activity, axis=0)
        sorted_indices = np.argsort(argmax_indices)
        im = ax.imshow(output_activity[:, sorted_indices].transpose(), aspect='equal', cmap='binary',interpolation="nearest")
        ax.set_xticks([0, num_output_units])
        ax.set_yticks([0, num_output_units])
        ax.set_xlabel('Pattern ID',fontsize=fontsize,labelpad=0)
        ax.set_ylabel('Output unit ID',fontsize=fontsize,labelpad=-2)
        ax.tick_params(labelsize=fontsize)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Output activity', rotation=270, labelpad=7, fontsize=fontsize)
        cbar.ax.tick_params(labelsize=fontsize)

    #Similarity matrix
    num_patterns = 2**num_units_history_dict['FF_Inh'][model_seed]['Input']
    for i,description in enumerate(description_list):
        ax = fig.add_subplot(axes[i, 2])
        similarity_matrix = similarity_matrix_history_dict[description][model_seed]['Output']
        im = ax.imshow(similarity_matrix, aspect='equal', cmap='viridis',vmin=0, vmax=1,interpolation="nearest")
        ax.set_xticks([0, num_output_units])
        ax.set_yticks([0, num_output_units])
        ax.set_xlabel('Pattern ID',fontsize=fontsize,labelpad=0)
        ax.set_ylabel('Pattern ID',fontsize=fontsize ,labelpad=-2)
        ax.tick_params(labelsize=fontsize)
        cbar = plt.colorbar(im, ax=ax, ticks=[0, 1])
        cbar.set_label('Cosine similarity', rotation=270, labelpad=7, fontsize=fontsize)
        cbar.ax.tick_params(labelsize=fontsize)

    #Cumulative distribution for sparsity (1 - fraction active)
    description_list = ['Input-Output-lognormal','FF_Inh','FF_Inh_no_sel_loss']
    label_dict['Input-Output-lognormal'] = 'No inhibition'

    ax = fig.add_subplot(axes[0, 3])
    cumulative_sparsity_dict = {}
    for description in description_list:
        cumulative_sparsity, mean_sparsity, cdf_prob_bins, SD = plot_cumulative_sparsity(sparsity_history_dict[description])
        cumulative_sparsity_dict[description] = cumulative_sparsity
        ax.plot(mean_sparsity, cdf_prob_bins, label=label_dict[description],color=color_dict[description])
        error_min = mean_sparsity - SD
        error_max = mean_sparsity + SD
        ax.fill_betweenx(cdf_prob_bins, error_min, error_max, alpha=0.2,color=color_dict[description])
    ax.tick_params(labelsize=fontsize)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xlabel('Sparsity',fontsize=fontsize,labelpad=0)
    ax.set_ylabel('Cumulative \nprobability',fontsize=fontsize,labelpad=0)

    s, p1 = stats.ks_2samp(cumulative_sparsity_dict['Input-Output-lognormal'],
                          cumulative_sparsity_dict['FF_Inh'])
    s, p2 = stats.ks_2samp(cumulative_sparsity_dict['Input-Output-lognormal'],
                          cumulative_sparsity_dict['FF_Inh_no_sel_loss'])
    s, p3 = stats.ks_2samp(cumulative_sparsity_dict['FF_Inh'],
                          cumulative_sparsity_dict['FF_Inh_no_sel_loss'])
    path_to_file = 'ks_tests.txt'
    mode = 'a' if os.path.exists(path_to_file) else 'w'
    with open(path_to_file, mode) as f:
        f.write("\nFig3 Sparsity Stats:"
                f"\nLognormal Vs FF Inh: p = {p1}"
                f"\nLognormal Vs FF No selectivity: p = {p2}"
                f"\nFF Inh Vs FF No selectivity: p = {p3}")

    #Cumulative distribution for selectivity
    ax = fig.add_subplot(axes[0, 4])
    cumulative_selectivity_dict = {}
    for description in description_list:
        cumulative_selectivity, mean_selectivity, cdf_prob_bins, SD = plot_cumulative_selectivity(selectivity_history_dict[description])
        cumulative_selectivity_dict[description] = cumulative_selectivity
        ax.plot(mean_selectivity, cdf_prob_bins, label=label_dict[description],color=color_dict[description])
        error_min = mean_selectivity - SD
        error_max = mean_selectivity + SD
        ax.fill_betweenx(cdf_prob_bins, error_min, error_max,alpha=0.2,color=color_dict[description])
    ax.tick_params(labelsize=fontsize)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xlabel('Selectivity',fontsize=fontsize,labelpad=0)
    ax.set_ylabel('Cumulative \nprobability',fontsize=fontsize,labelpad=0)
    ax.legend(loc='best',frameon=False,fontsize=fontsize,handlelength=1)

    s, p1 = stats.ks_2samp(cumulative_selectivity_dict['Input-Output-lognormal'],
                          cumulative_selectivity_dict['FF_Inh'])
    s, p2 = stats.ks_2samp(cumulative_selectivity_dict['Input-Output-lognormal'],
                          cumulative_selectivity_dict['FF_Inh_no_sel_loss'])
    s, p3 = stats.ks_2samp(cumulative_selectivity_dict['FF_Inh'],
                          cumulative_selectivity_dict['FF_Inh_no_sel_loss'])
    with open(path_to_file, 'a') as f:
        f.write("\nFig3 Selectivity Stats:"
                f"\nLognormal Vs FF Inh: p = {p1}"
                f"\nLognormal Vs FF No selectivity: p = {p2}"
                f"\nFF Inh Vs FF No selectivity: p = {p3}")

    #Cumulative distribution for discriminability
    ax = fig.add_subplot(axes[1, 3])
    cumulative_discriminability_dict = {}
    for description in description_list:
        cumulative_discriminability, mean_discriminability, cdf_prob_bins, SD = plot_cumulative_discriminability(similarity_matrix_history_dict[description])
        cumulative_discriminability_dict[description] = cumulative_discriminability
        ax.plot(mean_discriminability, cdf_prob_bins, label=label_dict[description],color=color_dict[description])
        error_min = mean_discriminability - SD
        error_max = mean_discriminability + SD
        ax.fill_betweenx(cdf_prob_bins, error_min, error_max,alpha=0.2,color=color_dict[description])
    ax.tick_params(labelsize=fontsize)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xlabel('Discriminability',fontsize=fontsize,labelpad=0)
    ax.set_ylabel('Cumulative \nprobability',fontsize=fontsize,labelpad=0)


    s, p1 = stats.ks_2samp(cumulative_discriminability_dict['Input-Output-lognormal'],
                          cumulative_discriminability_dict['FF_Inh'])
    s, p2 = stats.ks_2samp(cumulative_discriminability_dict['Input-Output-lognormal'],
                          cumulative_discriminability_dict['FF_Inh_no_sel_loss'])
    s, p3 = stats.ks_2samp(cumulative_discriminability_dict['FF_Inh'],
                          cumulative_discriminability_dict['FF_Inh_no_sel_loss'])
    with open(path_to_file, 'a') as f:
        f.write("\nFig3 Discriminability Stats:"
                f"\nLognormal Vs FF Inh: p = {p1}"
                f"\nLognormal Vs FF No selectivity: p = {p2}"
                f"\nFF Inh Vs FF No selectivity: p = {p3}\n")



    sns.despine()
    fig.savefig('figures/Figure3.svg', edgecolor='white', dpi=300, facecolor='white', transparent=True)
    # fig.savefig('figures/Figure3.png', edgecolor='white', dpi=300, facecolor='white', transparent=True)


def plot_figure4(num_units_history_dict,network_activity_history_dict, selectivity_history_dict, similarity_matrix_history_dict,
                 sparsity_history_dict,color_dict,label_dict,model_seed='1234'):

    mm = 1 / 25.4  # millimeters to inches
    fig = plt.figure(figsize=(180 * mm, 120 * mm))
    axes = gs.GridSpec(nrows=3, ncols=5,
                       left=0.05,right=0.98,
                       top = 0.95, bottom = 0.2,
                       wspace=0.8, hspace=1.2)

    fontsize = 8
    description_list = ['FB_Inh','FF_Inh+FB_Inh']
    label_dict['FF_Inh+FB_Inh'] = 'Inhibition: FF+FB'

    # Model diagrams & titles
    for i,description in enumerate(description_list):
        ax = fig.add_subplot(axes[i, 0])
        ax.axis('off')
        ax.set_title(label_dict[description], fontsize=fontsize)

    # Output activity for FB, FF+FB
    num_output_units = num_units_history_dict['FF_Inh'][model_seed]['Output']
    for i,description in enumerate(description_list):
        ax = fig.add_subplot(axes[i, 1])
        output_activity = network_activity_history_dict[description][model_seed]['Output']
        argmax_indices = np.argmax(output_activity, axis=0)
        sorted_indices = np.argsort(argmax_indices)
        im = ax.imshow(output_activity[:, sorted_indices].transpose(), aspect='equal', cmap='binary',interpolation="nearest")
        ax.set_xticks([0,num_output_units])
        ax.set_yticks([0,num_output_units])
        ax.set_xlabel('Pattern ID',fontsize=fontsize,labelpad=0)
        ax.set_ylabel('Output unit ID',fontsize=fontsize,labelpad=0)
        ax.tick_params(labelsize=fontsize)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Output activity', rotation=270, labelpad=7,fontsize=fontsize)
        cbar.ax.tick_params(labelsize=fontsize)

    #Similarity matrix
    num_patterns = 2**num_units_history_dict['FF_Inh'][model_seed]['Input']
    for i,description in enumerate(description_list):
        ax = fig.add_subplot(axes[i, 2])
        similarity_matrix = similarity_matrix_history_dict[description][model_seed]['Output']
        im = ax.imshow(similarity_matrix, aspect='equal', cmap='viridis',vmin=0, vmax=1,interpolation="nearest")
        ax.set_xticks([0, num_patterns])
        ax.set_yticks([0, num_patterns])
        ax.set_xlabel('Pattern ID',fontsize=fontsize,labelpad=0)
        ax.set_ylabel('Pattern ID',fontsize=fontsize,labelpad=-2)
        ax.tick_params(labelsize=fontsize)
        cbar = plt.colorbar(im, ax=ax, ticks=[0, 1])
        cbar.set_label('Cosine similarity', rotation=270, labelpad=7, fontsize=fontsize)
        cbar.ax.tick_params(labelsize=fontsize)


    #Cumulative distribution for sparsity (1 - fraction active)
    description_list = ['Input-Output-lognormal','FF_Inh','FB_Inh','FF_Inh+FB_Inh']
    label_dict['FF_Inh+FB_Inh'] = 'Inhibition: FF + FB'
    label_dict['Input-Output-lognormal'] = 'No inhibition'

    ax = fig.add_subplot(axes[0, 3])
    cumulative_sparsity_dict = {}
    for description in description_list:
        cumulative_sparsity, mean_sparsity, cdf_prob_bins, SD = plot_cumulative_sparsity(sparsity_history_dict[description])
        cumulative_sparsity_dict[description] = cumulative_sparsity
        ax.plot(mean_sparsity, cdf_prob_bins, label=label_dict[description],color=color_dict[description])
        error_min = mean_sparsity - SD
        error_max = mean_sparsity + SD
        ax.fill_betweenx(cdf_prob_bins, error_min, error_max, alpha=0.2,color=color_dict[description])
    ax.tick_params(labelsize=fontsize)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xlabel('Sparsity',fontsize=fontsize,labelpad=0)
    ax.set_ylabel('Cumulative \nprobability',fontsize=fontsize,labelpad=0)

    s, p1 = stats.ks_2samp(cumulative_sparsity_dict['Input-Output-lognormal'],
                          cumulative_sparsity_dict['FF_Inh'])
    s, p2 = stats.ks_2samp(cumulative_sparsity_dict['Input-Output-lognormal'],
                          cumulative_sparsity_dict['FB_Inh'])
    s, p3 = stats.ks_2samp(cumulative_sparsity_dict['Input-Output-lognormal'],
                          cumulative_sparsity_dict['FF_Inh+FB_Inh'])
    s, p4 = stats.ks_2samp(cumulative_sparsity_dict['FF_Inh'],
                          cumulative_sparsity_dict['FB_Inh'])
    s, p5 = stats.ks_2samp(cumulative_sparsity_dict['FF_Inh'],
                          cumulative_sparsity_dict['FF_Inh+FB_Inh'])
    s, p6 = stats.ks_2samp(cumulative_sparsity_dict['FB_Inh'],
                          cumulative_sparsity_dict['FF_Inh+FB_Inh'])
    path_to_file = 'ks_tests.txt'
    mode = 'a' if os.path.exists(path_to_file) else 'w'
    with open(path_to_file, mode) as f:
        f.write("\nFig4 Sparsity Stats:"
                f"\nLognormal Vs FF Inh: p = {p1}"
                f"\nLognormal Vs FB Inh: p = {p2}"
                f"\nLognormal Vs FF+FB Inh: p = {p3}"
                f"\nFF Inh Vs FB Inh: p = {p4}"
                f"\nFF Inh Vs FF+FB Inh: p = {p5}"
                f"\nFB Inh Vs FF+FB Inh: p = {p6}")

    #Cumulative distribution for selectivity
    ax = fig.add_subplot(axes[0, 4])
    cumulative_selectivity_dict = {}
    for description in description_list:
        cumulative_selectivity, mean_selectivity, cdf_prob_bins, SD = plot_cumulative_selectivity(selectivity_history_dict[description])
        cumulative_selectivity_dict[description] = cumulative_selectivity
        ax.plot(mean_selectivity, cdf_prob_bins, label=label_dict[description],color=color_dict[description])
        error_min = mean_selectivity - SD
        error_max = mean_selectivity + SD
        ax.fill_betweenx(cdf_prob_bins, error_min, error_max,alpha=0.2,color=color_dict[description])
    ax.tick_params(labelsize=fontsize)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xlabel('Selectivity',fontsize=fontsize,labelpad=0)
    ax.set_ylabel('Cumulative \nprobability',fontsize=fontsize)
    ax.legend(loc='best',frameon=False,fontsize=fontsize,handlelength=1)

    s, p1 = stats.ks_2samp(cumulative_selectivity_dict['Input-Output-lognormal'],
                           cumulative_selectivity_dict['FF_Inh'])
    s, p2 = stats.ks_2samp(cumulative_selectivity_dict['Input-Output-lognormal'],
                           cumulative_selectivity_dict['FB_Inh'])
    s, p3 = stats.ks_2samp(cumulative_selectivity_dict['Input-Output-lognormal'],
                           cumulative_selectivity_dict['FF_Inh+FB_Inh'])
    s, p4 = stats.ks_2samp(cumulative_selectivity_dict['FF_Inh'],
                           cumulative_selectivity_dict['FB_Inh'])
    s, p5 = stats.ks_2samp(cumulative_selectivity_dict['FF_Inh'],
                           cumulative_selectivity_dict['FF_Inh+FB_Inh'])
    s, p6 = stats.ks_2samp(cumulative_selectivity_dict['FB_Inh'],
                           cumulative_selectivity_dict['FF_Inh+FB_Inh'])
    with open(path_to_file, 'a') as f:
        f.write("\nFig4 Selectivity Stats:"
                f"\nLognormal Vs FF Inh: p = {p1}"
                f"\nLognormal Vs FB Inh: p = {p2}"
                f"\nLognormal Vs FF+FB Inh: p = {p3}"
                f"\nFF Inh Vs FB Inh: p = {p4}"
                f"\nFF Inh Vs FF+FB Inh: p = {p5}"
                f"\nFB Inh Vs FF+FB Inh: p = {p6}")

    #Cumulative distribution for discriminability
    ax = fig.add_subplot(axes[1, 3])
    cumulative_discriminability_dict = {}
    for description in description_list:
        cumulative_discriminability, mean_discriminability, cdf_prob_bins, SD = plot_cumulative_discriminability(similarity_matrix_history_dict[description])
        cumulative_discriminability_dict[description] = cumulative_discriminability
        ax.plot(mean_discriminability, cdf_prob_bins, label=label_dict[description],color=color_dict[description])
        error_min = mean_discriminability - SD
        error_max = mean_discriminability + SD
        ax.fill_betweenx(cdf_prob_bins, error_min, error_max,alpha=0.2,color=color_dict[description])
    ax.tick_params(labelsize=fontsize)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xlabel('Discriminability',fontsize=fontsize,labelpad=0)
    ax.set_ylabel('Cumulative \nprobability',fontsize=fontsize)

    s, p1 = stats.ks_2samp(cumulative_discriminability_dict['Input-Output-lognormal'],
                           cumulative_discriminability_dict['FF_Inh'])
    s, p2 = stats.ks_2samp(cumulative_discriminability_dict['Input-Output-lognormal'],
                           cumulative_discriminability_dict['FB_Inh'])
    s, p3 = stats.ks_2samp(cumulative_discriminability_dict['Input-Output-lognormal'],
                           cumulative_discriminability_dict['FF_Inh+FB_Inh'])
    s, p4 = stats.ks_2samp(cumulative_discriminability_dict['FF_Inh'],
                           cumulative_discriminability_dict['FB_Inh'])
    s, p5 = stats.ks_2samp(cumulative_discriminability_dict['FF_Inh'],
                           cumulative_discriminability_dict['FF_Inh+FB_Inh'])
    s, p6 = stats.ks_2samp(cumulative_discriminability_dict['FB_Inh'],
                           cumulative_discriminability_dict['FF_Inh+FB_Inh'])
    with open(path_to_file, 'a') as f:
        f.write("\nFig4 Discriminability Stats:"
                f"\nLognormal Vs FF Inh: p = {p1}"
                f"\nLognormal Vs FB Inh: p = {p2}"
                f"\nLognormal Vs FF+FB Inh: p = {p3}"
                f"\nFF Inh Vs FB Inh: p = {p4}"
                f"\nFF Inh Vs FF+FB Inh: p = {p5}"
                f"\nFB Inh Vs FF+FB Inh: p = {p6}\n")


    sns.despine()
    fig.savefig('figures/Figure4.svg', edgecolor='white', dpi=300, facecolor='white', transparent=True)
    # fig.savefig('figures/Figure4.png', edgecolor='white', dpi=300, facecolor='white', transparent=True)


def plot_figure5(num_units_history_dict,network_activity_history_dict, selectivity_history_dict, similarity_matrix_history_dict,
                 sparsity_history_dict,color_dict,label_dict,model_seed='1234'):

    mm = 1 / 25.4  # millimeters to inches
    fig = plt.figure(figsize=(180 * mm, 120 * mm))
    axes = gs.GridSpec(nrows=3, ncols=5,
                       left=0.05,right=0.98,
                       top = 0.95, bottom = 0.2,
                       wspace=0.8, hspace=1.2)
    fontsize = 8
    description_list = ['FF_Inh+indirect_FB_Inh','FF_Inh+indirect_FB_Inh_c','FF_Inh+indirect_FB_Inh+FB_Exc']
    num_output_units = num_units_history_dict['FF_Inh'][model_seed]['Output']

    # Model diagrams & titles
    for i,description in enumerate(description_list):
        ax = fig.add_subplot(axes[i, 0])
        ax.axis('off')
        ax.set_title(label_dict[description], fontsize=fontsize)

    # Output activity for FF, FB, FF+FB
    for i,description in enumerate(description_list):
        ax = fig.add_subplot(axes[i,1])
        output_activity = network_activity_history_dict[description][model_seed]['Output']
        argmax_indices = np.argmax(output_activity, axis=0)
        sorted_indices = np.argsort(argmax_indices)
        im = ax.imshow(output_activity[:, sorted_indices].transpose(), aspect='equal', cmap='binary',interpolation="nearest")
        ax.set_xticks([0, num_output_units])
        ax.set_yticks([0, num_output_units])
        ax.set_xlabel('Pattern ID',fontsize=fontsize,labelpad=0)
        ax.set_ylabel('Output unit ID',fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Output activity', rotation=270, labelpad=7,fontsize=fontsize)
        cbar.ax.tick_params(labelsize=fontsize)

    #Similarity matrix
    num_patterns = 2**num_units_history_dict['FF_Inh'][model_seed]['Input']

    for i,description in enumerate(description_list):
        ax = fig.add_subplot(axes[i, 2])
        similarity_matrix = similarity_matrix_history_dict[description][model_seed]['Output']
        im = ax.imshow(similarity_matrix, aspect='equal', cmap='viridis',vmin=0, vmax=1,interpolation="nearest")
        ax.set_xticks([0, num_patterns])
        ax.set_yticks([0, num_patterns])
        ax.set_xlabel('Pattern ID',fontsize=fontsize,labelpad=0)
        ax.set_ylabel('Pattern ID',fontsize=fontsize,labelpad=-2)
        ax.tick_params(labelsize=fontsize)
        cbar = plt.colorbar(im, ax=ax, ticks=[0,1])
        cbar.set_label('Cosine similarity', rotation=270, labelpad=7,fontsize=fontsize)
        cbar.ax.tick_params(labelsize=fontsize)

    #Cumulative distribution for sparsity (1 - fraction active)
    description_list = ['FF_Inh+FB_Inh','FF_Inh+indirect_FB_Inh','FF_Inh+indirect_FB_Inh_c','FF_Inh+indirect_FB_Inh+FB_Exc']
    label_dict['FF_Inh+FB_Inh'] = 'Inhibition: FF + direct FB'

    ax = fig.add_subplot(axes[0, 3])
    cumulative_sparsity_dict = {}
    for description in description_list:
        cumulative_sparsity, mean_sparsity, cdf_prob_bins, SD = plot_cumulative_sparsity(sparsity_history_dict[description])
        cumulative_sparsity_dict[description] = cumulative_sparsity
        ax.plot(mean_sparsity, cdf_prob_bins, label=label_dict[description],color=color_dict[description])
        error_min = mean_sparsity - SD
        error_max = mean_sparsity + SD
        ax.fill_betweenx(cdf_prob_bins, error_min, error_max, alpha=0.2,color=color_dict[description])
    ax.tick_params(labelsize=fontsize)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xlabel('Sparsity',fontsize=fontsize,labelpad=0)
    ax.set_ylabel('Cumulative \nprobability',fontsize=fontsize)

    s, p1 = stats.ks_2samp(cumulative_sparsity_dict['FF_Inh+FB_Inh'],
                           cumulative_sparsity_dict['FF_Inh+indirect_FB_Inh'])
    s, p2 = stats.ks_2samp(cumulative_sparsity_dict['FF_Inh+FB_Inh'],
                           cumulative_sparsity_dict['FF_Inh+indirect_FB_Inh_c'])
    s, p3 = stats.ks_2samp(cumulative_sparsity_dict['FF_Inh+FB_Inh'],
                           cumulative_sparsity_dict['FF_Inh+indirect_FB_Inh+FB_Exc'])
    s, p4 = stats.ks_2samp(cumulative_sparsity_dict['FF_Inh+indirect_FB_Inh'],
                           cumulative_sparsity_dict['FF_Inh+indirect_FB_Inh_c'])
    s, p5 = stats.ks_2samp(cumulative_sparsity_dict['FF_Inh+indirect_FB_Inh'],
                           cumulative_sparsity_dict['FF_Inh+indirect_FB_Inh+FB_Exc'])
    path_to_file = 'ks_tests.txt'
    mode = 'a' if os.path.exists(path_to_file) else 'w'
    with open(path_to_file, mode) as f:
        f.write("\nFig5 Sparsity Stats:"
                f"\ndirect FB Vs indirect FB: p = {p1}"
                f"\ndirect FB Vs (-)recurrent: p = {p2}"
                f"\ndirect FB Vs (+)FB Exc: p = {p3}"
                f"\nindirect FB Vs (-)recurrent: p = {p4}"
                f"\nindirect FB Vs (+)FB Exc: p = {p5}")

    #Cumulative distribution for selectivity
    ax = fig.add_subplot(axes[1, 3])
    cumulative_selectivity_dict = {}
    for description in description_list:
        cumulative_selectivity, mean_selectivity, cdf_prob_bins, SD = plot_cumulative_selectivity(selectivity_history_dict[description])
        cumulative_selectivity_dict[description] = cumulative_selectivity
        ax.plot(mean_selectivity, cdf_prob_bins, label=label_dict[description],color=color_dict[description])
        error_min = mean_selectivity - SD
        error_max = mean_selectivity + SD
        ax.fill_betweenx(cdf_prob_bins, error_min, error_max,alpha=0.2,color=color_dict[description])
    ax.tick_params(labelsize=fontsize)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xlabel('Selectivity',fontsize=fontsize,labelpad=0)
    ax.set_ylabel('Cumulative \nprobability',fontsize=fontsize)

    s, p1 = stats.ks_2samp(cumulative_selectivity_dict['FF_Inh+FB_Inh'],cumulative_selectivity_dict['FF_Inh+indirect_FB_Inh'])
    s, p2 = stats.ks_2samp(cumulative_selectivity_dict['FF_Inh+FB_Inh'],cumulative_selectivity_dict['FF_Inh+indirect_FB_Inh_c'])
    s, p3 = stats.ks_2samp(cumulative_selectivity_dict['FF_Inh+FB_Inh'],cumulative_selectivity_dict['FF_Inh+indirect_FB_Inh+FB_Exc'])
    s, p4 = stats.ks_2samp(cumulative_selectivity_dict['FF_Inh+indirect_FB_Inh'],cumulative_selectivity_dict['FF_Inh+indirect_FB_Inh_c'])
    s, p5 = stats.ks_2samp(cumulative_selectivity_dict['FF_Inh+indirect_FB_Inh'],cumulative_selectivity_dict['FF_Inh+indirect_FB_Inh+FB_Exc'])
    with open(path_to_file, 'a') as f:
        f.write("\nFig5 Selectivity Stats:"
                f"\ndirect FB Vs indirect FB: p = {p1}"
                f"\ndirect FB Vs (-)recurrent: p = {p2}"
                f"\ndirect FB Vs (+)FB Exc: p = {p3}"
                f"\nindirect FB Vs (-)recurrent: p = {p4}"
                f"\nindirect FB Vs (+)FB Exc: p = {p5}")


    #Cumulative distribution for discriminability
    ax = fig.add_subplot(axes[2, 3])
    cumulative_discriminability_dict = {}
    for description in description_list:
        cumulative_discriminability, mean_discriminability, cdf_prob_bins, SD = plot_cumulative_discriminability(similarity_matrix_history_dict[description])
        cumulative_discriminability_dict[description] = cumulative_discriminability
        ax.plot(mean_discriminability, cdf_prob_bins, label=label_dict[description],color=color_dict[description])
        error_min = mean_discriminability - SD
        error_max = mean_discriminability + SD
        ax.fill_betweenx(cdf_prob_bins, error_min, error_max,alpha=0.2,color=color_dict[description])
    ax.tick_params(labelsize=fontsize)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xlabel('Discriminability',fontsize=fontsize,labelpad=0)
    ax.set_ylabel('Cumulative \nprobability',fontsize=fontsize)
    ax.legend(loc='best',frameon=False,fontsize=fontsize,handlelength=1)

    s, p1 = stats.ks_2samp(cumulative_discriminability_dict['FF_Inh+FB_Inh'],
                           cumulative_discriminability_dict['FF_Inh+indirect_FB_Inh'])
    s, p2 = stats.ks_2samp(cumulative_discriminability_dict['FF_Inh+FB_Inh'],
                           cumulative_discriminability_dict['FF_Inh+indirect_FB_Inh_c'])
    s, p3 = stats.ks_2samp(cumulative_discriminability_dict['FF_Inh+FB_Inh'],
                           cumulative_discriminability_dict['FF_Inh+indirect_FB_Inh+FB_Exc'])
    s, p4 = stats.ks_2samp(cumulative_discriminability_dict['FF_Inh+indirect_FB_Inh'],
                           cumulative_discriminability_dict['FF_Inh+indirect_FB_Inh_c'])
    s, p5 = stats.ks_2samp(cumulative_discriminability_dict['FF_Inh+indirect_FB_Inh'],
                           cumulative_discriminability_dict['FF_Inh+indirect_FB_Inh+FB_Exc'])
    with open(path_to_file, 'a') as f:
        f.write("\nFig5 Discriminability Stats:"
                f"\ndirect FB Vs indirect FB: p = {p1}"
                f"\ndirect FB Vs (-)recurrent: p = {p2}"
                f"\ndirect FB Vs (+)FB Exc: p = {p3}"
                f"\nindirect FB Vs (-)recurrent: p = {p4}"
                f"\nindirect FB Vs (+)FB Exc: p = {p5}\n")

    sns.despine()
    fig.savefig('figures/Figure5.svg', edgecolor='white', dpi=300, facecolor='white', transparent=True)
    # fig.savefig('figures/Figure5.png', edgecolor='white', dpi=300, facecolor='white', transparent=True)


def plot_S1(num_units_history_dict,network_activity_history_dict,color_dict,label_dict,model_seed='1234'):
    'S1, related to Figure 3'

    mm = 1 / 25.4  # millimeters to inches
    fig = plt.figure(figsize=(180 * mm, 120 * mm))
    axes = gs.GridSpec(nrows=3, ncols=5,
                       left=0.05,right=0.98,
                       top = 0.95, bottom = 0.2,
                       wspace=0.8, hspace=1.2)


    fontsize = 8
    num_output_units = num_units_history_dict['FF_Inh'][model_seed]['Output']
    description_list = ['FF_Inh','FB_Inh','FF_Inh+FB_Inh']

    # Model diagrams & titles
    for i,description in enumerate(description_list):
        ax = fig.add_subplot(axes[i, 0])
        ax.axis('off')
        ax.set_title(label_dict[description], fontsize=fontsize)

    # Output activity for FF units
    ax = fig.add_subplot(axes[0, 1])
    output_activity = network_activity_history_dict['FF_Inh'][model_seed]['FF_Inh']
    num_units = num_units_history_dict['FF_Inh'][model_seed]['FF_Inh']
    argmax_indices = np.argmax(output_activity, axis=0)
    sorted_indices = np.argsort(argmax_indices)
    im = ax.imshow(output_activity[:, sorted_indices].transpose(), aspect='auto', cmap='binary',interpolation="nearest")
    ax.set_xticks([0, num_output_units])
    ax.set_yticks([-0.5, num_units - 0.5])
    ax.set_yticklabels([0, num_units])
    ax.set_xlabel('Pattern ID',fontsize=fontsize,labelpad=0)
    ax.set_ylabel('Output unit ID',fontsize=fontsize)
    ax.set_title('FF Inh',fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Output activity', rotation=270, labelpad=7,fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)

    # Output activity for FB units
    ax = fig.add_subplot(axes[1, 1])
    output_activity = network_activity_history_dict['FB_Inh'][model_seed]['FB_Inh']
    num_units = num_units_history_dict['FB_Inh'][model_seed]['FB_Inh']
    argmax_indices = np.argmax(output_activity, axis=0)
    sorted_indices = np.argsort(argmax_indices)
    im = ax.imshow(output_activity[:, sorted_indices].transpose(), aspect='auto', cmap='binary',interpolation="nearest")
    ax.set_xticks([0, num_output_units])
    ax.set_yticks([-0.5, num_units - 0.5])
    ax.set_yticklabels([0, num_units])
    ax.set_xlabel('Pattern ID',fontsize=fontsize,labelpad=0)
    ax.set_ylabel('Output unit ID',fontsize=fontsize)
    ax.set_title("FB Inh",fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Output activity', rotation=270, labelpad=7,fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)

    population_list = ['FF_Inh','FB_Inh']
    plot_titles = ['FF Inh','FB Inh']
    for i, population in enumerate(population_list):
        ax = fig.add_subplot(axes[2, i+1])
        output_activity = network_activity_history_dict['FF_Inh+FB_Inh'][model_seed][population]
        num_units = num_units_history_dict['FF_Inh+FB_Inh'][model_seed][population]
        argmax_indices = np.argmax(output_activity, axis=0)
        sorted_indices = np.argsort(argmax_indices)
        im = ax.imshow(output_activity.transpose()[sorted_indices, :], aspect='auto', cmap='binary',interpolation="nearest")
        ax.set_xticks([0, num_output_units])
        ax.set_yticks([-0.5, num_units - 0.5])
        ax.set_yticklabels([0, num_units])
        ax.set_xlabel('Pattern ID',fontsize=fontsize,labelpad=0)
        ax.set_ylabel('Output unit ID',fontsize=fontsize)
        ax.set_title(plot_titles[i],fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Output activity', rotation=270, labelpad=7,fontsize=fontsize)
        cbar.ax.tick_params(labelsize=fontsize)


    sns.despine()
    fig.savefig('figures/S1_F3.svg', edgecolor='white', dpi=300, facecolor='white', transparent=True)
    # fig.savefig('figures/S1_F3.png', edgecolor='white', dpi=300, facecolor='white', transparent=True)


def plot_S2(num_units_history_dict,network_activity_history_dict,color_dict,label_dict,model_seed='1234'):
    'S2, related to Figure 4'

    mm = 1 / 25.4  # millimeters to inches
    # fig = plt.figure(figsize=(180 * mm, 100 * mm))
    # axes = gs.GridSpec(nrows=3, ncols=3,
    #                    left=0.08,right=0.94,
    #                    top = 0.9, bottom = 0.06,
    #                    wspace=1, hspace=2)
    fig = plt.figure(figsize=(180 * mm, 120 * mm))
    axes = gs.GridSpec(nrows=3, ncols=5,
                       left=0.05,right=0.98,
                       top = 0.95, bottom = 0.2,
                       wspace=0.8, hspace=1.2)
    fontsize = 8

    description_list = ['FF_Inh+indirect_FB_Inh', 'FF_Inh+indirect_FB_Inh_c','FF_Inh+indirect_FB_Inh+FB_Exc']

    # Model diagrams & titles
    for i,description in enumerate(description_list):
        ax = fig.add_subplot(axes[i, 0])
        ax.axis('off')
        ax.set_title(label_dict[description], fontsize=fontsize)

    # Output activity for FF_Inh, FB_Inh, and FB_Exc units
    num_patterns = 2**num_units_history_dict['FF_Inh'][model_seed]['Input']
    population_list = ['FF_Inh','FB_Inh','FB_Exc']
    population_names = ['FF Inh','FB Inh','FB Exc']
    for row,description in enumerate(description_list):
        for col,population in enumerate(population_list):
            ax = fig.add_subplot(axes[row, col+1])
            output_activity = network_activity_history_dict[description][model_seed][population]
            argmax_indices = np.argmax(output_activity, axis=0)
            sorted_indices = np.argsort(argmax_indices)
            im = ax.imshow(output_activity[:, sorted_indices].transpose(), aspect='auto', cmap='binary',interpolation="nearest")
            num_units = num_units_history_dict[description][model_seed][population]
            ax.set_xticks([0, num_patterns])
            ax.set_yticks([-0.5, num_units-0.5])
            ax.set_yticklabels([0, num_units])
            ax.set_xlabel('Pattern ID',fontsize=fontsize,labelpad=0)
            ax.set_ylabel('Output unit ID',fontsize=fontsize)
            ax.set_title(population_names[col],fontsize=fontsize)
            ax.tick_params(labelsize=fontsize)
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Output activity', rotation=270, labelpad=7,fontsize=fontsize)
            cbar.ax.tick_params(labelsize=fontsize)

    sns.despine()
    fig.savefig('figures/S2_F4.svg', edgecolor='white', dpi=300, facecolor='white', transparent=True)
    # fig.savefig('figures/S2_F4.png', edgecolor='white', dpi=300, facecolor='white', transparent=True)


def plot_S3(network_activity_dynamics_history_dict,color_dict,label_dict,model_seed='1234'):
    'S4, related to Figure 4: population dynamics'
    mm = 1 / 25.4  # millimeters to inches
    # fig = plt.figure(figsize=(180 * mm, 100 * mm))
    # axes = gs.GridSpec(nrows=2, ncols=5,
    #                    left=0.08,right=0.98,
    #                    top = 0.9, bottom = 0.2,
    #                    wspace=1, hspace=1.2)
    fig = plt.figure(figsize=(180 * mm, 120 * mm))
    axes = gs.GridSpec(nrows=3, ncols=5,
                       left=0.1,right=0.98,
                       top = 0.95, bottom = 0.2,
                       wspace=0.5, hspace=1.2)

    fontsize = 8

    population_list = ['Output','FF_Inh','FB_Inh','FB_Exc']
    plot_titles = ['Output','FF Inh','FB Inh','FB Exc']
    description = 'FF_Inh+indirect_FB_Inh+FB_Exc'

    for i,population in enumerate(population_list):
        ax = fig.add_subplot(axes[0, i])

        mean_activity_array = []
        for model_seed in network_activity_dynamics_history_dict[description].keys():
            mean_across_patterns = np.mean(network_activity_dynamics_history_dict[description][model_seed][population],axis=0)
            mean_activity = np.mean(mean_across_patterns,axis=0)
            mean_activity_array.append(mean_activity)

            ax.plot(mean_activity, color=[0.8,0.8,0.8])

        mean_activity_array = np.array(mean_activity_array)
        mean_across_seeds = np.mean(mean_activity_array, axis=0)
        ax.plot(mean_across_seeds, color='red')
        ax.tick_params(labelsize=fontsize)
        ax.set_xticks(np.arange(0,400,100))
        ax.set_xlabel('Time (ms)',fontsize=fontsize)
        if i==0:
            ax.set_ylabel('Mean activity',fontsize=fontsize)
        ax.set_title(plot_titles[i],fontsize=fontsize)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.savefig('figures/S3_F4.svg', edgecolor='white', dpi=300, facecolor='white', transparent=True)
    # fig.savefig('figures/S3_F4.png', edgecolor='white', dpi=300, facecolor='white', transparent=True)

#############################################################################

@click.command()
@click.option("--data_file_path", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--dynamics_file_path", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--model_seed", type=str, default='1234')
@click.option("--plot", is_flag=True)

def main(data_file_path, dynamics_file_path, model_seed, plot):

    # Import data
    _,num_units_history_dict,_,_,weight_history_dict, network_activity_history_dict, sparsity_history_dict, \
        similarity_matrix_history_dict, selectivity_history_dict,fraction_active_patterns_history_dict,\
        _ = import_slice_data(data_file_path)
    network_activity_dynamics_history_dict =  import_dynamic_activity(dynamics_file_path)

    # Specify figure parameters (colors & labels)
    color_dict = {'Ideal': 'purple',
                  'Input': 'r',
                  'Input-Output-uniform': 'grey',
                  'Input-Output-lognormal': 'k',
                  'FF_Inh': 'cyan',
                  'FF_Inh_no_sel_loss': 'green',
                  'FB_Inh': 'orange',
                  'FF_Inh+FB_Inh': 'sienna',
                  'FF_Inh+indirect_FB_Inh': 'lime',
                  'FF_Inh+indirect_FB_Inh_c': 'magenta',
                  'FF_Inh+indirect_FB_Inh+FB_Exc': 'b'}

    label_dict = {'Ideal':'Ideal output',
                  'Input': 'Input',
                  'Input-Output-uniform': 'Uniform weights',
                  'Input-Output-lognormal': 'Log-normal weights',
                  'FF_Inh': 'FF Inhibition',
                  'FF_Inh_no_sel_loss': 'No selectivity constraint',
                  'FB_Inh': 'FB Inhibition',
                  'FF_Inh+FB_Inh': 'FF + FB Inhibition',
                  'FF_Inh+indirect_FB_Inh': 'FF + indirect FB Inhibition',
                  'FF_Inh+indirect_FB_Inh_b': 'FF_Inh+indirect_FB_Inh_b',
                  'FF_Inh+indirect_FB_Inh_c': '(-) FB Exc -> FB Exc',
                  'FF_Inh+indirect_FB_Inh+FB_Exc': '(+) FB Exc -> Output',
                  'FF_Inh+indirect_FB_Inh+FB_Exc_b': 'FB Exc -> Output +\n FB Inh  -> FB Exc'}

    # Generate figures
    plot_S3(network_activity_dynamics_history_dict,color_dict,label_dict)

    plot_S2(num_units_history_dict,network_activity_history_dict,color_dict,label_dict,model_seed)

    plot_figure5(num_units_history_dict,network_activity_history_dict, selectivity_history_dict, similarity_matrix_history_dict,
                 sparsity_history_dict,color_dict,label_dict,model_seed)

    plot_S1(num_units_history_dict,network_activity_history_dict,color_dict,label_dict,model_seed)

    plot_figure4(num_units_history_dict,network_activity_history_dict, selectivity_history_dict, similarity_matrix_history_dict,
                 sparsity_history_dict,color_dict,label_dict,model_seed)

    plot_figure3(num_units_history_dict, weight_history_dict, network_activity_history_dict, selectivity_history_dict, similarity_matrix_history_dict,
                 sparsity_history_dict,color_dict,label_dict,model_seed)

    plot_figure2(similarity_matrix_history_dict,num_units_history_dict,color_dict,label_dict,model_seed)

    plot_figure1(num_units_history_dict, sparsity_history_dict, selectivity_history_dict, similarity_matrix_history_dict,
                 weight_history_dict, network_activity_history_dict,color_dict,label_dict,model_seed)

    if plot:
        plt.show()

if __name__ == '__main__':
    # this extra flag stops the click module from forcing system exit when python is in interactive mode
    main(standalone_mode=False)
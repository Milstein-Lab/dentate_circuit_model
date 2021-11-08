# from nested.analyze import PopulationStorage
# storage = PopulationStorage()

# python Figure1.py --data_file_path='data/20211104_000741_dentate_optimization_2_exported_output_slice.hdf5'

import click
import numpy as np
from math import ceil
import h5py
from copy import deepcopy
import matplotlib.pyplot as plt
from optimize_dynamic_model import analyze_slice, get_binary_input_patterns

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style(style='ticks')

def import_slice_data(data_file_path, model_seed = 'all'):
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

            if isinstance(model_seed, str):
                if model_seed == 'all':
                    model_seed_list = list(f[description].keys())

            #     elif model_seed in f:
            #         model_seed_list = [model_seed]
            #     else:
            #         raise RuntimeError('import_model_data: model with seed: %s not found in %s' %
            #                            (model_seed, data_file_path))
            # elif isinstance(model_seed, Iterable):
            #     model_seed_list = list(model_seed)
            #     for model_seed in model_seed_list:
            #         if model_seed not in f:
            #             raise RuntimeError('import_model_data: model with seed: %s not found in %s' %
            #                                (model_seed, data_file_path))
            # else:
            #     raise RuntimeError('import_model_data: specify model model_seed as str or list of str')


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

        print('import_model_data: loaded data from {} for the following models: {}, {}'.format(\
            data_file_path, description, model_seed_list))


    return  model_config_history_dict, num_units_history_dict, activation_function_history_dict, \
            weight_config_history_dict, weight_history_dict, network_activity_history_dict, \
            sparsity_history_dict, similarity_matrix_history_dict, selectivity_history_dict, \
            fraction_active_patterns_history_dict, fraction_active_units_history_dict


def plot_figure1(num_units_history_dict, sparsity_history_dict, selectivity_history_dict, similarity_matrix_history_dict,
                 weight_history_dict, network_activity_history_dict, model_seed='seed:1234'):

    fig, axes = plt.subplots(3, 3, figsize=(6, 6))

    # Top left: input patterns
    num_input_units = num_units_history_dict['Input-Output-lognormal'][model_seed]['Input']
    sorted_input_patterns = (get_binary_input_patterns(num_input_units, sort=True)).transpose()

    im1 = axes[0, 0].imshow(sorted_input_patterns, aspect='auto',cmap='gray_r')
    axes[0, 0].set_xlabel('input pattern ID')
    axes[0, 0].set_ylabel('input unit ID')
    axes[0, 0].set_title('Input Patterns')
    cbar = plt.colorbar(im1, ax=axes[0, 0])

    # Top middle: simple input-output network diagram
    axes[0, 1].axis('off')
    # Top right: ideal output (identity matrix)
    num_output_units = num_units_history_dict['Input-Output-lognormal'][model_seed]['Output']
    im2 = axes[0, 2].imshow(np.eye(num_output_units), aspect='auto', cmap='viridis')
    axes[0, 2].set_xticks(np.arange(0, num_output_units+1, num_output_units / 4))
    axes[0, 2].set_yticks(np.arange(0, num_output_units + 1, num_output_units / 4))
    axes[0, 2].set_xlabel('output pattern ID')
    axes[0, 2].set_ylabel('output unit ID')
    axes[0, 2].set_title('Ideal Output Activity')
    cbar = plt.colorbar(im2, ax=axes[0, 2])

    # Middle left: weight distribution
    weight_dict_lognormal = weight_history_dict['Input-Output-lognormal'][model_seed]['Output']['Input']
    weight_dict_uniform = weight_history_dict['Input-Output-uniform'][model_seed]['Output']['Input']
    max_weight_lognormal = np.max(weight_dict_lognormal)
    max_weight_uniform = np.max(weight_dict_uniform)
    bin_width = max_weight_lognormal / 50
    hist, edges = np.histogram(weight_dict_lognormal,
                               bins=np.arange(-bin_width / 2., max_weight_lognormal + bin_width, bin_width), density=True)
    axes[1, 0].plot(edges[:-1] + bin_width / 2., hist * bin_width,
                    label='log-normal')
    hist, edges = np.histogram(weight_dict_uniform,
                               bins=np.arange(-bin_width / 2., max_weight_lognormal + bin_width, bin_width),
                               density=True)
    axes[1, 0].plot(edges[:-1] + bin_width / 2., hist * bin_width,
                    label='uniform')
    axes[1, 0].set_xlabel('Weights')
    axes[1, 0].set_title('Weight Distributions')
    axes[1, 0].legend(loc='best', frameon=False)


    # Middle middle: output activity uniform
    output_activity_uniform_dict = network_activity_history_dict['Input-Output-uniform'][model_seed]['Output']
    im5 = axes[1, 1].imshow(output_activity_uniform_dict.transpose(), aspect = 'auto', cmap='viridis')
    axes[1, 1].set_xticks(np.arange(0, num_output_units+1, num_output_units / 4))
    axes[1, 1].set_yticks(np.arange(0, num_output_units + 1, num_output_units / 4))
    axes[1, 1].set_xlabel('Output Pattern ID')
    axes[1, 1].set_ylabel('Output Unit ID')
    axes[1, 1].set_title('Output Activity (uniform)')
    cbar = plt.colorbar(im5, ax=axes[1, 1])

    # Middle right: output activity log-normal
    output_activity_lognormal_dict = network_activity_history_dict['Input-Output-lognormal'][model_seed]['Output']
    im6 = axes[1, 2].imshow(output_activity_lognormal_dict.transpose(), aspect = 'auto', cmap='viridis')
    axes[1, 2].set_xticks(np.arange(0, num_output_units+1, num_output_units / 4))
    axes[1, 2].set_yticks(np.arange(0, num_output_units + 1, num_output_units / 4))
    axes[1, 2].set_xlabel('Output Pattern ID')
    axes[1, 2].set_ylabel('Output Unit ID')
    axes[1, 2].set_title('Output Activity (log-normal)')
    cbar = plt.colorbar(im6, ax=axes[1, 2])

    # Bottom left: sparsity
    active_unit_count_lognormal = sparsity_history_dict['Input-Output-lognormal'][model_seed]['Output']
    active_unit_count_uniform = sparsity_history_dict['Input-Output-uniform'][model_seed]['Output']
    im3 = axes[2,0].scatter((np.arange(0, num_output_units)), active_unit_count_lognormal, label = 'log-normal')
    axes[2, 0].scatter((np.arange(0, num_output_units)), active_unit_count_uniform, label='uniform')
    x = [0,num_output_units]
    y = [1,1]
    axes[2, 0].plot(x,y, color = 'red',label='Ideal')
    axes[2, 0].set_xticks(np.arange(0, num_output_units+1, num_output_units / 4))
    axes[2, 0].set_yticks(np.arange(0, num_output_units + 1, num_output_units / 4))
    axes[2, 0].set_xlabel('input pattern ID')
    axes[2, 0].set_ylabel('# active neurons') #active output neurons count
    axes[2, 0].set_title('Sparsity')
    axes[2, 0].legend(loc='best', frameon=False)

    # Bottom middle: selectivity
    num_patterns_selected_lognormal = selectivity_history_dict['Input-Output-lognormal'][model_seed]['Output']
    num_patterns_selected_uniform = selectivity_history_dict['Input-Output-uniform'][model_seed]['Output']
    max_response = np.max(num_patterns_selected_lognormal)
    bin_width = max_response / 80
    hist, edges = np.histogram(num_patterns_selected_lognormal,
                               bins=np.arange(-bin_width / 2., max_response + bin_width, bin_width), density=True)
    axes[2, 1].plot(edges[:-1] + bin_width / 2., hist * bin_width,
                    label='log-normal')
    hist, edges = np.histogram(num_patterns_selected_uniform,
                               bins=np.arange(-bin_width / 2., max_response + bin_width, bin_width), density=True)
    axes[2, 1].plot(edges[:-1] + bin_width / 2., hist * bin_width,
                    label='uniform')
    x = [1, 1]
    y = [0, ceil(np.max(hist*bin_width)*10)/10] #set ideal line to same height as other distributions, rounded up
    axes[2, 1].plot(x, y, color='red', label='Ideal')
    axes[2, 1].set_xticks(np.arange(0, num_output_units+1, num_output_units / 4))
    axes[2, 1].set_xlabel('# patterns selected')
    axes[2, 1].set_title('Selectivity Distribution')
    axes[2, 1].legend(loc='best', frameon=False)

    # Bottom right: discriminability (cosine similarity)
    output_similarity_lognormal = similarity_matrix_history_dict['Input-Output-lognormal'][model_seed]['Output']
    output_similarity_uniform = similarity_matrix_history_dict['Input-Output-uniform'][model_seed]['Output']
    bin_width = 0.05
    invalid_indexes_lognormal = np.isnan(output_similarity_lognormal)
    hist, edges = np.histogram(output_similarity_lognormal[~invalid_indexes_lognormal],
                               bins=np.arange(-bin_width / 2., 1 + bin_width, bin_width), density=True)
    axes[2, 2].plot(edges[:-1] + bin_width / 2., hist * bin_width,
                    label='log-normal')
    invalid_indexes_uniform = np.isnan(output_similarity_uniform)
    hist, edges = np.histogram(output_similarity_uniform[~invalid_indexes_uniform],
                               bins=np.arange(-bin_width / 2., 1 + bin_width, bin_width), density=True)
    axes[2, 2].plot(edges[:-1] + bin_width / 2., hist * bin_width,
                    label='uniform')
    x = [0, 0]
    y = [0, ceil(np.max(hist*bin_width)*10)/10] #set ideal line to same height as other distributions, rounded up
    axes[2, 2].plot(x, y, color='red', label='Ideal')
    axes[2, 2].set_xticks(np.arange(0, 1.25, 1 / 4))
    axes[2, 2].set_xlabel('pattern cosine similarity')
    axes[2, 2].set_title('Discriminability')
    axes[2, 2].legend(loc='best', frameon=False)

    fig.suptitle('Figure 1')
    fig.tight_layout(w_pad=1.0, h_pad=2.0, rect=(0., 0., 1., 0.98))

    sns.despine()
    plt.show()
    # plt.savefig(data/fig1.png, edgecolor='black', dpi=400, facecolor='black', transparent=True)

    #TODO: 2)edit top-left 3)save fig1 & edit dimensions/spacing etc

def plot_cumulative_similarity(similarity_matrix_history_dict):
    cumulative_similarity = []
    n_bins = 100
    cdf_prob_bins = np.arange(1., n_bins + 1.) / n_bins

    for model_seed in similarity_matrix_history_dict:
        similarity_matrix = similarity_matrix_history_dict[model_seed]['Output']

        # extract all values below diagonal
        similarity_matrix_idx = np.tril_indices_from(similarity_matrix, -1)
        similarity = similarity_matrix[similarity_matrix_idx]

        #remove nan values
        invalid_idx = np.isnan(similarity)
        similarity = similarity[~invalid_idx]

        similarity = np.sort(similarity[:])
        quantiles = [np.quantile(similarity, pi) for pi in cdf_prob_bins]
        cumulative_similarity.append(quantiles)

    cumulative_similarity = np.array(cumulative_similarity)
    mean_similarity = np.mean(cumulative_similarity, axis=0)
    SD = np.std(cumulative_similarity, axis=0)
    SEM = SD / np.sqrt(cumulative_similarity.shape[0])

    return mean_similarity, cdf_prob_bins, SD


def plot_cumulative_selectivity(selectivity_history_dict):
    cumulative_selectivity = []
    n_bins = 100
    cdf_prob_bins = np.arange(1., n_bins + 1.) / n_bins

    for model_seed in selectivity_history_dict:
        selectivity = selectivity_history_dict[model_seed]['Output']

        selectivity = np.sort(selectivity[:])
        quantiles = [np.quantile(selectivity, pi) for pi in cdf_prob_bins]
        cumulative_selectivity.append(quantiles)

    cumulative_selectivity = np.array(cumulative_selectivity)
    mean_selectivity = np.mean(cumulative_selectivity, axis=0)

    SD = np.std(cumulative_selectivity, axis=0)
    SEM = SD / np.sqrt(cumulative_selectivity.shape[0])

    return mean_selectivity, cdf_prob_bins, SD

def plot_figure2(network_activity_history_dict, selectivity_history_dict, similarity_matrix_history_dict,
                 fraction_active_patterns_history_dict):
    #Make 3x3 figure

    #Top row: network diagrams for FF, FB, FF+FB

    #Middle row: activity heatmap for FF, FB, FF+FB

    #Bottom row: cumulative distributions for selectivity, similarity, fraction active

    fig, axes = plt.subplots(3, 3, figsize=(6, 6))


    mean_selectivity, cdf_prob_bins, SD = plot_cumulative_selectivity(selectivity_history_dict['FF_Inh'])
    axes[2,0].plot(mean_selectivity, cdf_prob_bins, color='red', label='description')
    error_min = mean_selectivity - SD
    error_max = mean_selectivity + SD
    axes[2,0].fill_betweenx(cdf_prob_bins, error_min, error_max,
                      facecolor="gray",  # The fill color
                      color='gray',  # The outline color
                      alpha=0.2)  # Transparency of the fill

    axes[2,0].legend(loc='best')
    axes[2,0].set_title('Cumulative histograms')
    axes[2,0].set_xlabel('selectivity')
    axes[2,0].set_ylabel('probability')


    mean_similarity, cdf_prob_bins, SD = plot_cumulative_similarity(similarity_matrix_history_dict['FF_Inh'])
    axes[2,1].plot(mean_similarity, cdf_prob_bins, color='red', label='description')
    error_min = mean_similarity - SD
    error_max = mean_similarity + SD
    axes[2,1].fill_betweenx(cdf_prob_bins, error_min, error_max,
                      facecolor="gray",  # The fill color
                      color='gray',  # The outline color
                      alpha=0.2)  # Transparency of the fill
    axes[2,1].legend(loc='best')
    axes[2,1].set_title('Cumulative histograms')
    axes[2,1].set_xlabel('cosine similarity')
    axes[2,1].set_ylabel('probability')


    sns.despine()
    plt.show()


def plot_figure3():
    #Indirect FB inh by MCs
    return


def plot_figure4():
    #Direct E + indirect FB inh by MCs
    return


#############################################################################

@click.command()
@click.option("--data_file_path", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--model_seed", type=str, default='all')

def main(data_file_path,model_seed):
    _,num_units_history_dict,_,_,weight_history_dict, network_activity_history_dict, sparsity_history_dict, \
        similarity_matrix_history_dict, selectivity_history_dict,fraction_active_patterns_history_dict,\
        _ = import_slice_data(data_file_path,model_seed)

    globals().update(locals())

    # plot_figure1(num_units_history_dict, sparsity_history_dict, selectivity_history_dict, similarity_matrix_history_dict,
    #              weight_history_dict, network_activity_history_dict)

    plot_figure2(network_activity_history_dict, selectivity_history_dict, similarity_matrix_history_dict,
                 fraction_active_patterns_history_dict)

    # plot_average_model_summary(network_activity_dict, sparsity_dict, similarity_matrix_dict,
    #                        selectivity_dict, description)
    #
    # plot_average_dynamics(t, median_sparsity_dynamics_dict, median_similarity_dynamics_dict,
    #               mean_selectivity_dynamics_dict, fraction_nonzero_response_dynamics_dict, description)


if __name__ == '__main__':
    # this extra flag stops the click module from forcing system exit when python is in interactive mode
    main(standalone_mode=False)
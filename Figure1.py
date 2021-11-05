# from nested.analyze import PopulationStorage
# storage = PopulationStorage()

# python Figure1.py --data_file_path='data/20211104_000741_dentate_optimization_2_exported_output_slice.hdf5'

import click
import numpy as np
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
    fraction_nonzero_history_dict = {}

    # This clause evokes a "Context Manager" and takes care of opening and closing the file so we don't forget
    with h5py.File(data_file_path, 'r') as f:
        description = list(f.keys())[0]
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
                fraction_nonzero_dict = analyze_slice(network_activity_dict)

            model_config_history_dict[model_seed] = deepcopy(model_config_dict)
            num_units_history_dict[model_seed] = deepcopy(num_units_dict)
            activation_function_history_dict[model_seed] = deepcopy(activation_function_dict)
            weight_config_history_dict[model_seed] = deepcopy(weight_config_dict)
            weight_history_dict[model_seed] = deepcopy(weight_dict)
            network_activity_history_dict[model_seed] = deepcopy(network_activity_dict)

            sparsity_history_dict[model_seed] = sparsity_dict
            similarity_matrix_history_dict[model_seed] = similarity_matrix_dict
            selectivity_history_dict[model_seed] = selectivity_dict
            fraction_nonzero_history_dict[model_seed] = fraction_nonzero_dict

    print('import_model_data: loaded data from %s for the following model model_seeds: %s' %
          (data_file_path, model_seed_list))

    return  model_config_history_dict, num_units_history_dict, activation_function_history_dict, \
            weight_config_history_dict, weight_history_dict, network_activity_history_dict, \
            sparsity_history_dict, similarity_matrix_history_dict, selectivity_history_dict, \
            fraction_nonzero_history_dict


def plot_figure1(num_units_history_dict, sparsity_history_dict, selectivity_history_dict, similarity_matrix_history_dict,
                 weight_history_dict, network_activity_history_dict):

    fig, axes = plt.subplots(3, 3, figsize=(6, 6))

    # Top left: input patterns
    num_input_units = num_units_history_dict['seed:1234']['Input']
    sorted_input_patterns = (get_binary_input_patterns(num_input_units, sort=True)).transpose()

    im1 = axes[0, 0].imshow(sorted_input_patterns, aspect='auto',cmap='gray_r')
    axes[0, 0].set_xlabel('Input Pattern ID')
    axes[0, 0].set_ylabel('Input Unit ID')
    axes[0, 0].set_title('Input Patterns')
    cbar = plt.colorbar(im1, ax=axes[0, 0])
    # Top middle: simple input-output network diagram

    # Top right: ideal output (identity matrix)
    num_output_units = num_units_history_dict['seed:1234']['Output']
    im2 = axes[0, 2].imshow(np.eye(num_output_units), aspect='auto', cmap='viridis')
    axes[0, 2].set_xticks(np.arange(0, num_output_units+1, num_output_units / 4))
    axes[0, 2].set_yticks(np.arange(0, num_output_units + 1, num_output_units / 4))
    axes[0, 2].set_xlabel('Output Pattern ID')
    axes[0, 2].set_ylabel('Output Unit ID')
    axes[0, 2].set_title('Ideal Output Activity')
    cbar = plt.colorbar(im2, ax=axes[0, 2])

    # Middle left: weight distribution

    # Middle middle: output activity uniform

    # Middle right: output activity log-normal

    # Bottom left: sparsity
    active_output_unit_count = sparsity_history_dict['seed:1234']['Output']
    im3 = axes[2,0].scatter((np.arange(0, num_output_units)), active_output_unit_count, label = 'log-normal')
    x = [0,num_output_units]
    y = [1,1]
    axes[2, 0].plot(x,y, color = 'red',label='Ideal')
    axes[2, 0].set_xticks(np.arange(0, num_output_units+1, num_output_units / 4))
    axes[2, 0].set_yticks(np.arange(0, num_output_units + 1, num_output_units / 4))
    axes[2, 0].set_xlabel('Input Pattern ID')
    axes[2, 0].set_ylabel('# of active neurons') #active output neurons count
    axes[2, 0].set_title('Sparsity')
    axes[2, 0].legend(loc='best', frameon=False)

    # Bottom middle: selectivity
    num_patterns_selected = selectivity_history_dict['seed:1234']['Output']
    max_response = np.max(num_patterns_selected)
    bin_width = max_response / 20
    hist, edges = np.histogram(num_patterns_selected,
                               bins=np.arange(-bin_width / 2., max_response + bin_width, bin_width), density=True)
    axes[2, 1].plot(edges[:-1] + bin_width / 2., hist * bin_width,
                    label='log-normal')
    x = [1, 1]
    y = [0, 1]
    axes[2, 1].plot(x, y, color='red', label='Ideal')
    axes[2, 1].set_xticks(np.arange(0, num_output_units+1, num_output_units / 4))
    axes[2, 1].set_xlabel('# of patterns selected')
    axes[2, 1].set_title('Selectivity Distribution')
    axes[2, 1].legend(loc='best', frameon=False)

    # Bottom right: discriminability (cosine similarity)
    output_similarity = similarity_matrix_history_dict['seed:1234']['Output']
    bin_width = 0.05
    invalid_indexes = np.isnan(output_similarity)
    hist, edges = np.histogram(output_similarity[~invalid_indexes],
                               bins=np.arange(-bin_width / 2., 1 + bin_width, bin_width), density=True)
    axes[2, 2].plot(edges[:-1] + bin_width / 2., hist * bin_width,
                    label='log-normal')
    x = [0, 0]
    y = [0, 1]
    axes[2, 2].plot(x, y, color='red', label='Ideal')
    axes[2, 2].set_xticks(np.arange(0, 1.25, 1 / 4))
    axes[2, 2].set_xlabel('Output Pattern Cosine Similarity')
    axes[2, 2].set_title('Discriminability')
    axes[2, 2].legend(loc='best', frameon=False)

    fig.suptitle('Figure 1')
    fig.tight_layout(w_pad=1.0, h_pad=2.0, rect=(0., 0., 1., 0.98))

    sns.despine()
    plt.show()
    # plt.savefig(file.jpeg, edgecolor='black', dpi=400, facecolor='black', transparent=True)

def plot_cumulative_similarity(similarity_matrix_history_dict):

    cumulative_similarity = []
    bin_width = 0.01
    for model_seed in similarity_matrix_history_dict:
        similarity_matrix = similarity_matrix_history_dict[model_seed]['Output']

        # extract all values below diagonal
        similarity_matrix_idx = np.tril_indices_from(similarity_matrix, -1)
        similarity_array = similarity_matrix[similarity_matrix_idx]

        invalid_indexes = np.isnan(similarity_array)
        hist, edges = np.histogram(similarity_array[~invalid_indexes],
                                   bins=np.arange(-bin_width / 2., 1 + bin_width, bin_width), density=True)

        cumulative_similarity.append(np.cumsum(hist) * bin_width)

    cumulative_similarity = np.array(cumulative_similarity)
    mean_similarity = np.mean(cumulative_similarity, axis=0)

    fig, ax = plt.subplots(figsize=(8, 4))

    edges = edges[:-1] + bin_width / 2  # plot line using center of each bin (instead of edges)

    SEM = np.std(cumulative_similarity, axis=0)  # /np.sqrt(cumulative_similarity.shape[0])
    error_min = edges - SEM
    error_max = edges + SEM

    plt.fill_betweenx(mean_similarity, error_min, error_max,
                      facecolor="gray",  # The fill color
                      color='gray',  # The outline color
                      alpha=0.2)  # Transparency of the fill

    plt.plot(edges, mean_similarity, color='red', label='FF_I')

    ax.legend(loc='best')
    ax.set_title('Cumulative histograms')
    ax.set_xlabel('cosine similarity')
    ax.set_ylabel('probability')

    sns.despine()
    plt.show()


def plot_cumulative_selectivity(selectivity_history_dict):

    cumulative_selectivity = []
    bin_width = 0.01
    max_value = 128 #maximum number of possible responses per neuron
    for model_seed in selectivity_history_dict:
        selectivity = selectivity_history_dict[model_seed]['Output']

        hist, edges = np.histogram(selectivity,
                           bins=np.arange(-bin_width / 2., max_value+bin_width, bin_width), density=True)

        cumulative_selectivity.append(np.cumsum(hist) * bin_width)

    cumulative_selectivity = np.array(cumulative_selectivity)
    mean_selectivity = np.mean(cumulative_selectivity, axis=0)

    fig, ax = plt.subplots(figsize=(8, 4))

    edges = edges[:-1] + bin_width / 2  # plot line using center of each bin (instead of edges)

    SEM = np.std(cumulative_selectivity, axis=0)  # /np.sqrt(cumulative_similarity.shape[0])
    error_min = edges - SEM
    error_max = edges + SEM

    plt.fill_betweenx(mean_selectivity, error_min, error_max,
                      facecolor="gray",  # The fill color
                      color='gray',  # The outline color
                      alpha=0.2)  # Transparency of the fill

    plt.plot(edges, mean_selectivity, color='red', label='FF_I')

    ax.legend(loc='best')
    ax.set_title('Cumulative histograms')
    ax.set_xlabel('selectivity')
    ax.set_ylabel('probability')

    sns.despine()
    plt.show()

def plot_figure2():
    return
    # FF Inh
        #E->E
        #E->I
        #I->E

    # FF+FB Inh
        #E->E
        #E->I_ff
        #E->I_fb

    #plot weight distributions
    #plot...

    similarity = []
    for model_seed in similarity_matrix_history_dict:
        similarity_matrix = similarity_matrix_history_dict[model_seed]['Output']

        # extract all values below diagonal
        similarity_matrix_idx = np.tril_indices_from(similarity_matrix, -1)
        similarity_array = similarity_matrix[similarity_matrix_idx]

        similarity.append(similarity_array)



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
        similarity_matrix_history_dict, selectivity_history_dict,_ = import_slice_data(data_file_path,model_seed)

    globals().update(locals())

    plot_figure1(num_units_history_dict, sparsity_history_dict, selectivity_history_dict, similarity_matrix_history_dict,
                 weight_history_dict, network_activity_history_dict)

    # plot_figure2(similarity_matrix_history_dict)
    # plot_cumulative_selectivity(selectivity_history_dict)

    # plot_average_model_summary(network_activity_dict, sparsity_dict, similarity_matrix_dict,
    #                        selectivity_dict, description)
    #
    # plot_average_dynamics(t, median_sparsity_dynamics_dict, median_similarity_dynamics_dict,
    #               mean_selectivity_dynamics_dict, fraction_nonzero_response_dynamics_dict, description)


if __name__ == '__main__':
    # this extra flag stops the click module from forcing system exit when python is in interactive mode
    main(standalone_mode=False)
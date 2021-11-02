# from export_from_hdf5 import import_model_data
# import_model_data('data/20211101_232302_dentate_optimization_4_exported_output.hdf5')

import h5py
from copy import deepcopy

def import_model_data(data_file_path,model_seed='all'):
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
    similarity_matrix_history_dict = {}

    # This clause evokes a "Context Manager" and takes care of opening and closing the file so we don't forget
    with h5py.File(data_file_path, 'r') as f:
        if isinstance(model_seed, str):
            if model_seed == 'all':
                model_seed = list(f.keys())
            elif model_seed in f:
                model_seed_list = [model_seed]
            else:
                raise RuntimeError('import_model_data: model with seed: %s not found in %s' %
                                   (model_seed, data_file_path))
        elif isinstance(model_seed, Iterable):
            model_seed_list = list(model_seed)
            for model_seed in model_seed_list:
                if model_seed not in f:
                    raise RuntimeError('import_model_data: model with seed: %s not found in %s' %
                                       (model_seed, data_file_path))
        else:
            raise RuntimeError('import_model_data: specify model model_seed as str or list of str')

        for model_seed in model_seed_list:
            model_config_dict = {}
            num_units_dict = {}
            activation_function_dict = {}
            weight_config_dict = {}
            weight_dict = {}
            network_activity_dict = {}

            model_group = f[model_seed]
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
                    # load the meta data for the weight configuration of this projection
                    for key, value in group[post_pop][pre_pop].attrs.items():
                        weight_config_dict[post_pop][pre_pop][key] = value

            group = model_group['activity']
            for post_pop in group:
                network_activity_dict[post_pop] = group[post_pop][:]
                num_units_dict[post_pop] = group[post_pop].attrs['num_units']
                if 'activation_function' in group[post_pop].attrs:
                    activation_function_dict[post_pop] = \
                        get_callable_from_str(group[post_pop].attrs['activation_function'])

            model_config_history_dict[model_seed] = deepcopy(model_config_dict)
            num_units_history_dict[model_seed] = deepcopy(num_units_dict)
            activation_function_history_dict[model_seed] = deepcopy(activation_function_dict)
            weight_config_history_dict[model_seed] = deepcopy(weight_config_dict)
            weight_history_dict[model_seed] = deepcopy(weight_dict)
            network_activity_history_dict[model_seed] = deepcopy(network_activity_dict)

    print('import_model_data: loaded data from %s for the following model model_seeds: %s' %
          (data_file_path, model_seed_list))

    return model_config_history_dict, num_units_history_dict, activation_function_history_dict, \
           weight_config_history_dict, weight_history_dict, network_activity_history_dict



def plot_model_summary(network_activity_dict, similarity_matrix_dict, model_seed=None):
    """
    Generate a panel of plots summarizing the activity of each layer
    :param network_activity_dict: dict:
        {'population label': 2d array of float (number of input patterns, number of units in this population)
        }
    :param similarity_matrix_dict: dict:
        {'population label': 2d array of float (number of input patterns, number of input patterns)
        }
    :param model_seed: str
    """
    num_of_populations = len(network_activity_dict)

    fig, axes = plt.subplots(4, num_of_populations, figsize=(3.5 * num_of_populations, 12))
    for i, population in enumerate(network_activity_dict):
        im1 = axes[0, i].imshow(network_activity_dict[population], aspect='auto')
        cbar = plt.colorbar(im1, ax=axes[0, i])
        cbar.ax.set_ylabel('Unit activity', rotation=270, labelpad=20)
        axes[0, i].set_xlabel('Unit ID')
        axes[0, i].set_ylabel('Input pattern ID')
        axes[0, i].set_title('Activity\n%s population' % population)

        axes[1, i].scatter(range(len(network_activity_dict[population])),
                           np.sum(network_activity_dict[population], axis=1))
        axes[1, i].set_xlabel('Input pattern ID')
        axes[1, i].set_ylabel('Summed population activity')
        axes[1, i].set_title('Summed activity\n%s population' % population)
        axes[1, i].spines["top"].set_visible(False)
        axes[1, i].spines["right"].set_visible(False)

        im2 = axes[2, i].imshow(similarity_matrix_dict[population], aspect='auto')
        axes[2, i].set_xlabel('Input pattern ID')
        axes[2, i].set_ylabel('Input pattern ID')
        axes[2, i].set_title('Similarity\n%s population' % population)
        plt.colorbar(im2, ax=axes[2, i])

        bin_width = 0.05
        hist, edges = np.histogram(similarity_matrix_dict[population],
                                   bins=np.arange(-bin_width / 2., 1 + bin_width, bin_width), density=True)
        axes[3, i].plot(edges[:-1] + bin_width / 2., hist * bin_width)
        axes[3, i].set_xlabel('Cosine similarity')
        axes[3, i].set_ylabel('Probability')
        axes[3, i].set_title('Pairwise similarity distribution\n%s population' % population)
        axes[3, i].spines["top"].set_visible(False)
        axes[3, i].spines["right"].set_visible(False)

    if model_seed is not None:
        fig.suptitle(model_seed)
    fig.tight_layout(w_pad=3, h_pad=3, rect=(0., 0., 1., 0.98))
    fig.show()


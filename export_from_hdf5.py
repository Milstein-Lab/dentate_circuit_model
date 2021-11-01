import h5py
from copy import deepcopy



def import_model_data(data_file_path, description=None):
    """
    Imports model data from specified model configurations stored in the specified hdf5 file into nested dictionaries.
    If description is None, the list of model descriptions found in the file are printed.
    If description is 'all', all models found in the file are loaded and returned.
    If description is a valid str or list of str, only data from those model configurations will be imported and
    returned.
    :param data_file_path: str (path); path to hdf5 file
    :param description: str or list of str; unique identifiers for model configurations, used as keys in hdf5 file
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
            model_config_dict = {}
            num_units_dict = {}
            activation_function_dict = {}
            weight_config_dict = {}
            weight_dict = {}
            network_activity_dict = {}

            model_group = f[description]
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

            model_config_history_dict[description] = deepcopy(model_config_dict)
            num_units_history_dict[description] = deepcopy(num_units_dict)
            activation_function_history_dict[description] = deepcopy(activation_function_dict)
            weight_config_history_dict[description] = deepcopy(weight_config_dict)
            weight_history_dict[description] = deepcopy(weight_dict)
            network_activity_history_dict[description] = deepcopy(network_activity_dict)

    print('import_model_data: loaded data from %s for the following model descriptions: %s' %
          (data_file_path, description_list))

    return model_config_history_dict, num_units_history_dict, activation_function_history_dict, \
           weight_config_history_dict, weight_history_dict, network_activity_history_dict

def plot_model_summary(network_activity_dict, similarity_matrix_dict, description=None):
    """
    Generate a panel of plots summarizing the activity of each layer
    :param network_activity_dict: dict:
        {'population label': 2d array of float (number of input patterns, number of units in this population)
        }
    :param similarity_matrix_dict: dict:
        {'population label': 2d array of float (number of input patterns, number of input patterns)
        }
    :param description: str
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

    if description is not None:
        fig.suptitle(description)
    fig.tight_layout(w_pad=3, h_pad=3, rect=(0., 0., 1., 0.98))
    fig.show()


# from nested.analyze import PopulationStorage
# storage = PopulationStorage()

# python Figure1.py --data_file_path='data/20211104_000741_dentate_optimization_2_exported_output_slice.hdf5'

import click
import numpy as np
import h5py
from copy import deepcopy
import matplotlib.pyplot as plt
from optimize_dynamic_model import analyze_slice, get_binary_input_patterns


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


def plot_figure1(sparsity_history_dict, selectivity_history_dict, similarity_matrix_history_dict,
                 weight_history_dict, network_activity_history_dict):

    fig, axes = plt.subplots(3, 3, figsize=(6, 6),)

    # Top left: input patterns
    sorted_input_patterns = get_binary_input_patterns(num_input_units, sort=True)

    axes[0, 0].imshow(sorted_input_patterns, aspect='auto')

    # Top middle: simple input-output network diagram

    # Top right: ideal output (identity matrix)

    # Middle left: sparsity

    # Middle middle: selectivity

    # Middle right: discriminability (cosine similarity)

    # Bottom left: weight distribution

    # Bottom middle:

    # Bottom right: output activity


    plt.show()
    # plt.savefig(file.jpeg, edgecolor='black', dpi=400, facecolor='black', transparent=True)


def plot_figure2(similarity_matrix_history_dict):
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
    _,_,_,_,weight_history_dict, network_activity_history_dict, sparsity_history_dict, \
        similarity_matrix_history_dict, selectivity_history_dict, \
        fraction_nonzero_history_dict = import_slice_data(data_file_path,model_seed)

    # plot_figure1(sparsity_history_dict, selectivity_history_dict, similarity_matrix_history_dict,
    #              weight_history_dict, network_activity_history_dict)

    plot_figure2(similarity_matrix_history_dict)

    # plot_average_model_summary(network_activity_dict, sparsity_dict, similarity_matrix_dict,
    #                        selectivity_dict, description)
    #
    # plot_average_dynamics(t, median_sparsity_dynamics_dict, median_similarity_dynamics_dict,
    #               mean_selectivity_dynamics_dict, fraction_nonzero_response_dynamics_dict, description)


    globals().update(locals())


if __name__ == '__main__':
    # this extra flag stops the click module from forcing system exit when python is in interactive mode
    main(standalone_mode=False)
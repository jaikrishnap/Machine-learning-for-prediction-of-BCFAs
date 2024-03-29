import argparse
import math
import os
import pickle
import sys

import pandas as pd
from keras.models import load_model


def parse_input(file: str) -> pd.DataFrame:
    if file == '-':
        # Read from standard input
        file = sys.stdin

    df = pd.read_csv(file)
    return df


def write_output(data: pd.DataFrame, out_file: str):
    if out_file == '-':
        # Use standard output
        out_file = sys.stdout.buffer

    data.to_csv(out_file, index=False, mode='wb')


def get_input_features(data: pd.DataFrame) -> pd.DataFrame:
    necessary_features = {'wavelength', 'fractal_dimension', 'fraction_of_coating'}
    optional_features = {'number_of_primary_particles', 'vol_equi_radius_outer', 'vol_equi_radius_inner',
                         'equi_mobility_dia'}
    all_features = necessary_features.union(optional_features)
    all_features_list = ['wavelength', 'fractal_dimension', 'fraction_of_coating', 'primary_particle_size',
                         'number_of_primary_particles', 'vol_equi_radius_outer', 'vol_equi_radius_inner',
                         'equi_mobility_dia']

    # Check if all input features given
    # TODO: Right now, the code cannot handle missing features in some rows
    if all_features.issubset(data.columns):
        return data[all_features_list]

    # Check if necessary features are present
    if not necessary_features.issubset(data.columns):
        raise RuntimeError(f"Please specify mandatory feature values for {necessary_features}!")
    if data[necessary_features].isna().any(axis=None):
        raise RuntimeError(f"Please make sure to specify values for {necessary_features} in every row!")
    if len(optional_features.intersection(data.columns)) == 0:
        raise RuntimeError(f"Please make sure to specify at least one of {optional_features} in every row!")

    # Define some constants for conversions
    # TODO: Our code assumes that a pure BC primary particle has a fixed radius of 15 nm. We should probably relax
    #       this assumption and make this feature mandatory
    inner_pp_size = 15
    mob_dia_exp = 0.51
    mob_dia_factor = 10 ** (-2 * mob_dia_exp + 0.92)

    # Check if number of primary particles present in input
    if "primary_particle_size" not in data.columns:
        data['primary_particle_size'] = inner_pp_size / (1 - data['fraction_of_coating'] * 0.01) ** (1 / 3)

    # Compute missing features
    if 'number_of_primary_particles' not in data.columns:
        if 'equi_mobility_dia' in data.columns:
            data['number_of_primary_particles'] = (data['equi_mobility_dia'] /
                                                   (2 * mob_dia_factor * data['primary_particle_size'])) ** (1 / mob_dia_exp)
        else:
            if 'vol_equi_radius_outer' not in data.columns:
                data['vol_equi_radius_outer'] = data['vol_equi_radius_inner'] / \
                                                (1 - data['fraction_of_coating'] * 0.01) ** (1 / 3)
            data['number_of_primary_particles'] = (data['vol_equi_radius_outer'] / data['primary_particle_size']) ** 3

    if "equi_mobility_dia" not in data.columns:
        data['equi_mobility_dia'] = 2 * mob_dia_factor * data['primary_particle_size'] * data['number_of_primary_particles'] ** mob_dia_exp
    if "vol_equi_radius_outer" not in data.columns:
        data['vol_equi_radius_outer'] = data['number_of_primary_particles'] ** (1 / 3) * data['primary_particle_size']
    if "vol_equi_radius_inner" not in data.columns:
        data['vol_equi_radius_inner'] = (1 - data['fraction_of_coating'] * 0.01) ** (1 / 3) * data['vol_equi_radius_outer']

    return data[all_features_list]


def predict_krr(model_file: str, inputs: pd.DataFrame) -> pd.DataFrame:
    # Load model from file
    if model_file is None:
        model_file = os.path.join('models', 'krr_model.pkl')
    try:
        with open(model_file, mode='rb') as f:
            model = pickle.load(f)
    except Exception:
        # TODO: specify exceptions and handle
        raise

    if not isinstance(model, dict) or len({'shift', 'transform', 'regressor'}.intersection(model.keys())) < 3:
        raise RuntimeError('Loaded model file does not have the expected format! Expected a dictionary with keys'
                           '{shift, transform, regressor}. Please make sure that you specified the correct file!')

    # Convert input to a numpy array
    inputs = inputs.to_numpy(copy=True)

    # Add box cox shift to make inputs strictly positive
    inputs += model['shift']
    # Apply box-cox transformation with learned parameters
    inputs = model['transform'].transform(inputs)
    # Run KRR
    pred = model['regressor'].predict(inputs)
    pred = pd.DataFrame(data=pred, columns=['q_abs', 'q_sca', 'g'])

    return pred


def predict_ann(model_file: str, inputs: pd.DataFrame) -> pd.DataFrame:
    # Load model from file
    if model_file is None:
        model_file = os.path.join('models', 'ann_model.hdf5')
    try:
        # load neural network
        model = load_model(model_file)
        # load transformation
        with open(f'{os.path.splitext(model_file)[0]}.pkl', mode='rb') as f:
            transform_data = pickle.load(f)
    except Exception:
        # TODO: specify exceptions and handle
        raise

    if not isinstance(transform_data, dict) or len({'shift', 'transform'}.intersection(transform_data.keys())) < 2:
        raise RuntimeError('Loaded transform file does not have the expected format! Expected a dictionary with keys'
                           '{shift, transform}. Please make sure that you specified the correct file!')

    # Convert input to a numpy array
    inputs = inputs.to_numpy(copy=True)

    # Add box cox shift to make inputs strictly positive
    inputs += transform_data['shift']
    # Apply box-cox transformation with learned parameters
    inputs = transform_data['transform'].transform(inputs)
    # Run KRR
    pred = model.predict(inputs)
    pred = pd.DataFrame(data=pred, columns=['q_abs', 'q_sca', 'g'])

    return pred


def main(args: argparse.Namespace):
    data = parse_input(args.infile)
    inputs = get_input_features(data)

    if args.model.lower() == 'krr':
        prediction = predict_krr(args.model_file, inputs)
    elif args.model.lower() == 'ann':
        prediction = predict_ann(args.model_file, inputs)
    else:
        raise NotImplementedError(f'Model {args.model} is not implemented/supported!')

    # Compute remaining features
    data_out = pd.concat([inputs, prediction], join='inner', axis=1)

    data_out['mie_epsilon'] = 2
    data_out['length_scale_factor'] = 2 * math.pi / data_out['wavelength']

    # Add refractive indices
    ref_ind = pd.read_csv(args.refractive_indices, index_col='wavelength')
    data_out = data_out.join(ref_ind, how='left', on='wavelength')

    data_out['volume_total'] = 4 / 3 * math.pi * data_out['vol_equi_radius_outer'] ** 3
    data_out['volume_bc'] = 4 / 3 * math.pi * data_out['vol_equi_radius_inner'] ** 3
    data_out['volume_organics'] = data_out['volume_total'] - data_out['volume_bc']

    data_out['density_bc'] = 1.5  # Check
    data_out['density_organics'] = 1.1  # Check

    data_out['mass_bc'] = data_out['volume_bc'] * data_out['density_bc'] * 1e-21
    data_out['mass_organics'] = data_out['volume_organics'] * data_out['density_organics'] * 1e-21
    data_out['mass_total'] = data_out['mass_bc'] + data_out['mass_organics']
    data_out['mr_total/bc'] = data_out['mass_total'] / data_out['mass_bc']
    data_out['mr_nonBC/BC'] = data_out['mass_organics'] / data_out['mass_bc']

    data_out['q_ext'] = data_out['q_abs'] + data_out['q_sca']
    data_out['c_geo'] = math.pi * data_out['vol_equi_radius_outer'] ** 2
    data_out['c_ext'] = data_out['q_ext'] * data_out['c_geo'] * 1e-6
    data_out['c_abs'] = data_out['q_abs'] * data_out['c_geo'] * 1e-6
    data_out['c_sca'] = data_out['q_sca'] * data_out['c_geo'] * 1e-6
    data_out['ssa'] = data_out['q_sca'] / data_out['q_ext']
    data_out['mac_total'] = data_out['c_abs'] / data_out['mass_total'] * 1e-12
    data_out['mac_bc'] = data_out['c_abs'] / data_out['mass_bc'] * 1e-12
    data_out['mac_organics'] = data_out['c_abs'] / data_out['mass_organics'] * 1e-12

    # Reindex dataset to make sure columns are in the intended order
    data_out = data_out[['wavelength', 'fractal_dimension', 'fraction_of_coating', 'primary_particle_size',
                         'number_of_primary_particles', 'vol_equi_radius_outer', 'vol_equi_radius_inner',
                         'equi_mobility_dia', 'mie_epsilon', 'length_scale_factor', 'm_real_bc', 'm_im_bc',
                         'm_real_organics', 'm_im_organics', 'volume_total', 'volume_bc', 'volume_organics',
                         'density_bc', 'density_organics', 'mass_total', 'mass_organics', 'mass_bc', 'mr_total/bc',
                         'mr_nonBC/BC', 'q_ext', 'q_abs', 'q_sca', 'g', 'c_geo', 'c_ext', 'c_abs', 'c_sca', 'ssa',
                         'mac_total', 'mac_bc', 'mac_organics']]

    write_output(data_out, args.out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script to predict optical properties of black carbon fractal '
                                                 'aggregates from their physical properties.')
    parser.add_argument('-o', '--out-file', type=str, default='-',
                        help='Specify the path to save the result to. Currently, only the CSV format is supported. '
                             'Use "-" to write to STDOUT in CSV format. Please see the README file for more '
                             'information about the columns in the output.')
    parser.add_argument('-m', '--model', type=str, default='krr', choices=['krr', 'ann'],
                        help='Select the model to use for predicting the optical properties. Current choices are "krr" '
                             'for the Kernel Ridge Regression model and "ann" for the Neural Network.')
    parser.add_argument('--model-file', type=str, default=None,
                        help='Specifies the file from which to load the model. Defaults to the files in the "models" '
                             'folder.')
    parser.add_argument('-r', '--refractive-indices', type=str, default=os.path.join('data', 'refractive_indices.csv'),
                        help='Specifies the file from which to load the refractive indices for BC and organics for'
                             'each wavelength. Please see the README file for the correct format. Defaults to the'
                             'indices provided in data/refractive_indices.csv.')
    parser.add_argument('infile', type=str,
                        help='Specify the file to load the input data (physical properties) from. Please see the '
                             'README file for possible formats. Setting this to "-" reads from STDIN in CSV format.')

    parsed_args = parser.parse_args()
    main(parsed_args)

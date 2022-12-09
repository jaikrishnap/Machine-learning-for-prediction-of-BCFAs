import os
import pickle

import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import PowerTransformer


def main():
    df = pd.read_csv('database_new.csv')
    x = df[['wavelength', 'fractal_dimension', 'fraction_of_coating', 'primary_particle_size',
            'number_of_primary_particles', 'vol_equi_radius_outer', 'vol_equi_radius_inner',
            'equi_mobility_dia']].to_numpy(copy=True)
    y = df[['q_abs', 'q_sca', 'g']].to_numpy(copy=True)
    del df

    pt = PowerTransformer(method='box-cox')
    shift = 1e-10
    x_transformed = pt.fit_transform(x + shift)

    regressor = KernelRidge(alpha=0.0001, gamma=0.75, kernel='rbf')
    model = regressor.fit(x_transformed, y)
    with open(os.path.join('models', 'krr_model.pkl'), 'wb') as f:
        pickle.dump(dict(shift=shift, transform=pt, regressor=model), f)


if __name__ == "__main__":
    main()

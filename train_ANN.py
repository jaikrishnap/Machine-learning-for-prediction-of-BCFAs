import os
import pickle

import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Input
from keras.models import Sequential
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

    with open(os.path.join('models', 'ann_model.pkl'), 'wb') as f:
        pickle.dump(dict(shift=shift, transform=pt), f)

    model = Sequential()
    model.add(Input(shape=(8,)))
    for j in range(0, 8):
        model.add(Dense(512, kernel_initializer='he_normal', activation='relu'))
    model.add(Dense(3, kernel_initializer='he_normal', activation='linear'))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])

    checkpoint = ModelCheckpoint(os.path.join('models', 'ann_model.hdf5'), verbose=1, monitor='val_loss',
                                 save_best_only=True, mode='auto')
    es = EarlyStopping(monitor='val_loss', patience=100, verbose=1)
    callback_list = [checkpoint, es]
    _ = model.fit(x_transformed, y, epochs=1000, batch_size=32, validation_split=0.2, callbacks=callback_list)


if __name__ == "__main__":
    main()

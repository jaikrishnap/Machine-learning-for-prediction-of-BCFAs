import pandas as pd
from sklearn.preprocessing import PowerTransformer
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint
def main():
    df = pd.read_excel('database_new.xlsx')
    X = df.iloc[:, :8]
    Y = df.iloc[:, 25:28]

    pt = PowerTransformer(method='box-cox')
    X_transformed = pt.fit_transform(X + 0.00000000001)

    model = Sequential()
    model.add(Input(shape=(8,)))
    for j in range(0, 8):
        model.add(Dense(512, kernel_initializer='he_normal', activation='relu'))
    model.add(Dense(3, kernel_initializer='he_normal', activation='linear'))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])

    checkpoint = ModelCheckpoint('ann_model.hdf5', verbose=1, monitor='val_loss', save_best_only=True,
                                 mode='auto')
    es = EarlyStopping(monitor='val_loss', patience=100, verbose=1)
    callback_list = [checkpoint, es]
    history = model.fit(X_transformed, Y, epochs=1000, batch_size=32, validation_split=0.2, callbacks=callback_list)


if __name__ == "__main__":
    main()

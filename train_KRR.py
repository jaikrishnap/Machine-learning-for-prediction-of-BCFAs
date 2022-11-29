import pandas as pd
from sklearn.preprocessing import PowerTransformer
from sklearn.kernel_ridge import KernelRidge
import pickle
def main():
    df = pd.read_excel('database_new.xlsx')
    X = df.iloc[:, :8]
    Y = df.iloc[:, 25:28]

    pt = PowerTransformer(method='box-cox')
    X_transformed = pt.fit_transform(X + 0.00000000001)

    regressor = KernelRidge(alpha=0.0001, gamma=0.75, kernel='rbf')
    model = regressor.fit(X_transformed, Y)
    with open('krr_model.pkl', 'wb') as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    main()
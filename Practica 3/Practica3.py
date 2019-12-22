import pandas as pd
import numpy as np
from sklearn import preprocessing
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

if __name__ == "__main__":
    le = preprocessing.LabelEncoder()

    '''
    lectura de datos
    '''
    # los ficheros .csv se han preparado previamente para sustituir ,, y "Not known" por NaN (valores perdidos)
    X_train = pd.read_csv('nepal_earthquake_tra.csv')
    Y_train = pd.read_csv('nepal_earthquake_labels.csv')
    X_test = pd.read_csv('nepal_earthquake_tst.csv')

    # se quitan las columnas que no se usan
    X_train.drop(labels=['building_id'], axis=1, inplace=True)
    X_test.drop(labels=['building_id'], axis=1, inplace=True)
    Y_train.drop(labels=['building_id'], axis=1, inplace=True)
    """
    Esta linea muestra informacion sobre el dataset, la informacion esta en el archivo
    data_des.txt
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print (data_x.describe())
    """
    mask = X_train.isnull()
    X_train_tmp = X_train.fillna(9999)
    X_train_tmp = X_train_tmp.astype(str).apply(preprocessing.LabelEncoder().fit_transform)
    X_train_nan = X_train_tmp.where(~mask, X_train)

    mask = X_test.isnull()  # máscara para luego recuperar los NaN
    X_test_tmp = X_test.fillna(9999)  # LabelEncoder no funciona con NaN, se asigna un valor no usado
    X_test_tmp = X_test_tmp.astype(str).apply(preprocessing.LabelEncoder().fit_transform)  # se convierten categóricas en numéricas
    X_test_nan = X_test_tmp.where(~mask, X_test)  # se recuperan los NaN

    X = X_train_nan.values
    X_tst = X_test_nan.values
    y = np.ravel(Y_train.values)

    rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
    feat_selector = BorutaPy(rf, max_iter=9, n_estimators=200, verbose=0, random_state=123456)
    feat_selector.fit(X, y)

        




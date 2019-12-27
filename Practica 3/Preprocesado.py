import pandas as pd
import numpy as np
from sklearn import preprocessing
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import CondensedNearestNeighbour
import matplotlib.pyplot as plt

seed = 10
np.random.seed(seed)
img_path = "./Imagenes"
data_path = "./Datos"


def dataset_info(X_train, y, name):
    # Correlacion de las variables

    f = plt.figure(figsize=(19, 15))
    plt.matshow(X_train.corr(), fignum=f.number)
    cb = plt.colorbar()

    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix {}'.format(name), fontsize=16)
    plt.savefig("{}/Correlation-{}".format(img_path, name))

    plt.show()

    counts = np.unique(y, return_counts=True)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(counts[0], counts[1], width=0.8, align='center')
    ax.set_xticks(counts[0])
    ax.set_xticklabels(counts[0])
    plt.xlabel("Clase")
    plt.ylabel("Numero de ocurrencias")
    plt.title("Distribucion de clases {}".format(name))
    plt.savefig("{}/Distribucion-{}".format(img_path, name))
    plt.show()

    """
    Esta linea muestra informacion sobre las variables numéricas dataset, la informacion esta en el archivo
    data_des.txt. Esta comentado porque la salida no se ve muy bien desde el interprete
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print (data_x.describe())
    """


def categorical_to_number(x_training, x_test, y_training):
    # Extreamos las categoricas
    categorical_columns = list(x_training.select_dtypes('object').astype(str))
    categorical_features = x_training[categorical_columns]
    # Las eliminamos
    x_training = x_training.drop(columns=categorical_columns)
    # Aplicamos el label encoder
    tra_cat_enc = categorical_features.apply(preprocessing.LabelEncoder().fit_transform)
    # juntamos el conjunto de train
    procesed_x_tra = pd.concat((x_training, tra_cat_enc), axis=1, join='outer', ignore_index=False, keys=None,
                               levels=None, names=None, verify_integrity=False, copy=True)

    # Extreamos las categoricas
    categorical_columns = list(x_test.select_dtypes('object').astype(str))
    categorical_features = x_test[categorical_columns]
    # Las eliminamos
    x_test = x_test.drop(columns=categorical_columns)
    # Aplicamos el label encoder
    test_cat_enc = categorical_features.apply(preprocessing.LabelEncoder().fit_transform)
    # juntamos el conjunto de test
    procesed_x_test = pd.concat((x_test, test_cat_enc), axis=1, join='outer', ignore_index=False, keys=None,
                               levels=None, names=None, verify_integrity=False, copy=True)

    y = np.ravel(y_training)

    return procesed_x_tra,procesed_x_test, y


def sample_dataset(X_train, y_train):
    sampling_method = CondensedNearestNeighbour(random_state=seed)
    X_resampled, y_resampled = sampling_method.fit_resample(X_train, y_train)
    print("Shape del dataset original: {}. Shape del dataset procesado{} ".format(X_train.shape, X_resampled.shape))
    return X_resampled, y_resampled


if __name__ == "__main__":
    le = preprocessing.LabelEncoder()

    '''
    lectura de datos
    '''
    # los ficheros .csv se han preparado previamente para sustituir ,, y "Not known" por NaN (valores perdidos)
    X_tra = pd.read_csv('{}/nepal_earthquake_tra.csv'.format(data_path))
    Y_train = pd.read_csv('{}/nepal_earthquake_labels.csv'.format(data_path))
    X_tst = pd.read_csv('{}/nepal_earthquake_tst.csv'.format(data_path))

    # se quitan las columnas que no se usan
    X_tra.drop(labels=['building_id'], axis=1, inplace=True)
    X_tst.drop(labels=['building_id'], axis=1, inplace=True)
    Y_train.drop(labels=['building_id'], axis=1, inplace=True)

    print("Pasando categoricas a numericas")
    X, X_test_cat, y = categorical_to_number(X_tra, X_tst, Y_train)
    dataset_info(X, y, "original")
    X_train = pd.DataFrame(data=preprocessing.normalize(X.values), columns=list(X_tra.columns))
    X_test = pd.DataFrame(preprocessing.normalize(X_test_cat.values), columns=list(X_tst.columns))
    # Realizamos una tecnica de oversampling y despues una de undersampling
    print("Aplicando sampling")
    # X, y = sample_dataset(X, y)
    # Seleccion de caracteristicas

    data_frame_X_train = pd.DataFrame(data=X, columns=list(X_train.columns))
    data_frame_y_train = pd.DataFrame(data=y, columns=list(Y_train.columns))
    data_frame_X_test = pd.DataFrame(data=X_test_cat, columns=list(X_test.columns))
    # Printamos la info
    dataset_info(data_frame_X_train, data_frame_y_train, "sampled")
    # Exportamos a CSV
    data_frame_X_train.to_csv("{}/X_train_procesado.csv".format(data_path), index=False)
    data_frame_y_train.to_csv("{}/y_train_procesado.csv".format(data_path), index=False)
    data_frame_X_test.to_csv("{}/X_test_procesado.csv".format(data_path), index=False)

    """
    rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
    feat_selector = BorutaPy(rf, max_iter=9, n_estimators=200, verbose=0, random_state=seed)
    feat_selector.fit(X_sampled, y_sampled)
    X_sampled = feat_selector.transform(X_sampled)
    X_test_cat = feat_selector.transform(X_test_cat)

    data_frame_X_train = pd.DataFrame(data=X_sampled, columns=list(X_train.columns))
    data_frame_y_train = pd.DataFrame(data=y_sampled, columns=list(Y_train.columns))
    data_frame_X_test = pd.DataFrame(data=X_test_cat, columns=list(X_test.columns))

    data_frame_X_train.to_csv("{}/X_train_procesado_boruta.csv".format(data_path))
    data_frame_y_train.to_csv("{}/y_train_procesado_boruta.csv".format(data_path))
    data_frame_X_test.to_csv("{}/X_test_procesado_boruta.csv".format(data_path))
    """

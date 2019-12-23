import pandas as pd
import numpy as np
from sklearn import preprocessing
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from imblearn.combine import SMOTETomek
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
    mask = x_training.isnull()
    X_train_tmp = x_training.fillna(9999)
    X_train_tmp = X_train_tmp.astype(str).apply(preprocessing.LabelEncoder().fit_transform)
    X_train_nan = X_train_tmp.where(~mask, x_training)

    mask = x_test.isnull()  # máscara para luego recuperar los NaN
    X_test_tmp = x_test.fillna(9999)  # LabelEncoder no funciona con NaN, se asigna un valor no usado

    X_test_tmp = X_test_tmp.astype(str).apply(
        preprocessing.LabelEncoder().fit_transform)  # se convierten categóricas en numéricas

    X_test_nan = X_test_tmp.where(~mask, x_test)  # se recuperan los NaN

    X = X_train_nan
    X_tst = X_test_nan
    y = np.ravel(y_training)

    return X, X_tst, y


def sample_dataset(X_train, y_train):
    smote_tomek = SMOTETomek(random_state=seed)
    X_resampled, y_resampled = smote_tomek.fit_resample(X_train, y_train)
    print("Shape del dataset original: {}. Shape del dataset procesado{} ".format(X_train.shape, X_resampled.shape))
    return X_resampled, y_resampled


if __name__ == "__main__":
    le = preprocessing.LabelEncoder()

    '''
    lectura de datos
    '''
    # los ficheros .csv se han preparado previamente para sustituir ,, y "Not known" por NaN (valores perdidos)
    X_train = pd.read_csv('{}/nepal_earthquake_tra.csv'.format(data_path))
    Y_train = pd.read_csv('{}/nepal_earthquake_labels.csv'.format(data_path))
    X_test = pd.read_csv('{}/nepal_earthquake_tst.csv'.format(data_path))

    # se quitan las columnas que no se usan
    X_train.drop(labels=['building_id'], axis=1, inplace=True)
    X_test.drop(labels=['building_id'], axis=1, inplace=True)
    Y_train.drop(labels=['building_id'], axis=1, inplace=True)

    print ("Pasando categoricas a numericas")
    X, X_test_cat, y = categorical_to_number(X_train, X_test, Y_train)
    dataset_info(X, y, "original")
    # Realizamos una tecnica de oversampling y despues una de undersampling
    print("Aplicando SmoteTomek")
    X_sampled, y_sampled = sample_dataset(X, y)
    # Seleccion de caracteristicas


    data_frame_X_train = pd.DataFrame(data=X_sampled, columns=list(X_train.columns))
    data_frame_y_train = pd.DataFrame(data=y_sampled, columns=list(Y_train.columns))
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
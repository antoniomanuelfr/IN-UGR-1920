# -*- coding: utf-8 -*-

import time

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import sklearn.cluster as cluster
from sklearn import metrics
from sklearn import preprocessing
# explicitly require this experimental feature
from sklearn.experimental import enable_iterative_imputer  # noqa
# now you can import normally from sklearn.impute
from sklearn.impute import IterativeImputer
import sklearn.neighbors

"""
imputo iterative imputer para imputar valores perdidos. Esta opcion esta en experimental; 
https://scikit-learn.org/stable/modules/impute.html
"""
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from math import floor
import seaborn as sns


def norm_to_zero_one(df):
    return (df - df.min()) * 1.0 / (df.max() - df.min())


def generate_table(res):
    print("Algoritmo ; Tiempo; Calinski; Silhouette")
    for algoritmo in res:
        print("{}; {}s; {}; {} ".format(algoritmo, res[algoritmo][1], res[algoritmo][2], res[algoritmo][3]))


def exec_case(x_normal, algorithms):
    result = {'K Means': [], 'Mean Shift': []}

    for algorithm in algorithms:
        print('----- Ejecutando {}'.format(algorithm[0]))

        t = time.time()

        cluster_predict = algorithm[1].fit_predict(x_normal)
        tiempo = time.time() - t

        result[algorithm[0]].append(cluster_predict)
        result[algorithm[0]].append(tiempo)

        metric_ch = metrics.calinski_harabaz_score(x_normal, cluster_predict)
        result[algorithm[0]].append(metric_ch)

        # el cálculo de Silhouette puede consumir mucha RAM. Si son muchos datos, digamos más de 10k,
        # se puede seleccionar una muestra, p.ej., el 20%
        if len(x_normal) > 10000:
            muestra_silhoutte = 0.2
        else:
            muestra_silhoutte = 1.0

        metric_sc = metrics.silhouette_score(x_normal, cluster_predict, metric='euclidean',
                                             sample_size=floor(muestra_silhoutte * len(x_normal)), random_state=123456)
        result[algorithm[0]].append(metric_sc)

        # se convierte la asignación de clusters a DataFrame
        clusters = pd.DataFrame(cluster_predict, index=x_normal.index, columns=['cluster'])

        print("Tamaño de cada cluster usando {} :".format(algorithm[0]))
        size = clusters['cluster'].value_counts()
        for num, i in size.iteritems():
            print('%s: %5d (%5.2f%%)' % (num, i, 100 * i / len(clusters)))

        # Saco figuras
        print("---------- Preparando el heat map...")

        centers = pd.DataFrame(algorithm[1].cluster_centers_, columns=list(x_normal))
        centers_desnormal = centers.copy()

        # se convierten los centros a los rangos originales antes de normalizar
        for var in list(centers):
            centers_desnormal[var] = x_normal[var].min() + centers[var] * (x_normal[var].max() - x_normal[var].min())

        sns_plot = sns.heatmap(centers, cmap="YlGnBu", annot=centers_desnormal, fmt='.3f').get_figure()
        # sns_plot.savefig(algorithm[0] + "_heatmap.png")
        plt.show()

        print("---------- Preparando el scatter matrix...")
        # se añade la asignación de clusters como columna a x
        x_kmeans = pd.concat([x_normal, clusters], axis=1)
        sns.set()
        variables = list(x_kmeans)
        variables.remove('cluster')
        sns_plot = sns.pairplot(x_kmeans, vars=variables, hue="cluster", palette='Paired', plot_kws={"s": 25},
                                diag_kind="hist")  # en hue indicamos que la columna 'cluster' define los colores
        sns_plot.fig.subplots_adjust(wspace=.03, hspace=.03)
        # sns_plot.savefig(algorithm[0] + ".png")
        plt.show()
        print("")
        # '''
    generate_table(result)
    return result


if __name__ == "__main__":
    censo = pd.read_csv('mujeres_fecundidad_INE_2018.csv')

    # Se pueden reemplazar los valores desconocidos por un número
    # censo = censo.replace(np.NaN,0)

    # O imputar, por ejemplo con la> media
    k_means = cluster.KMeans(init='k-means++', n_clusters=5, n_init=5, n_jobs=10)
    ms = cluster.MeanShift(n_jobs=10)
    algoritmos = (('K Means', k_means), ('Mean Shift', ms))

    for col in censo:
        censo[col].fillna(censo[col].mean(), inplace=True)

    # Caso 1
    subset = censo.loc[(censo['EDAD'] > 19) & (censo['EDAD'] <= 30) & (censo['NHIJOS'] > 0)]
    # seleccionar casos
    # seleccionar variables de interés para clustering
    usadas = ['INGRESOS', 'NTRABA', 'ESTUDIOSA', 'NDESEOHIJO', 'PCUIDADOHIJOS']
    X = subset[usadas]
    X_normal = X.apply(norm_to_zero_one)

    exec_case(X_normal, algoritmos)

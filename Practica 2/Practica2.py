# -*- coding: utf-8 -*-

import time
from math import floor

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn.base
import sklearn.cluster as cluster
from sklearn import metrics


def norm_to_zero_one(df):
    return (df - df.min()) * 1.0 / (df.max() - df.min())


def generate_table(res):
    print("Algoritmo ; Tiempo; Calinski; Silhouette")
    for algoritmo in res:
        print("{}; {:.2f} s; {:.2f}; {:.2f}".format(algoritmo, res[algoritmo][1], res[algoritmo][2], res[algoritmo][3]))


def exec_case(x_normal, algorithms, save_figs=False, plot_figs=False):
    result = {}
    for alg in algorithms:
        result[alg[0]] = []

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
        if plot_figs:
            if algorithm[0] == 'Mean-Shift' or algorithm[0] == 'K-Means':
                print("---------- Preparando el heat map...")

                cluster_centers = algorithm[1].cluster_centers_
                centers = pd.DataFrame(cluster_centers, columns=list(x_normal))

                centers_desnormal = centers.copy()

                # se convierten los centros a los rangos originales antes de normalizar
                for var in list(centers):
                    centers_desnormal[var] = x_normal[var].min() + centers[var] * (
                            x_normal[var].max() - x_normal[var].min())

                sns_heatmap = sns.heatmap(centers, annot=centers_desnormal, fmt='.3f',
                                          cbar_kws={"orientation": "horizontal"}) \
                    .set_title(
                    "Heat map using {}".format(algorithm[0])).get_figure()
                plt.show()

                if save_figs:
                    sns_heatmap.savefig("Imagenes/{}_heatmap_{}.png".format(algorithm[0], cnt))

            print("---------- Preparando el scatter matrix...")
            # se añade la asignación de clusters como columna a x
            x_kmeans = pd.concat([x_normal, clusters], axis=1)
            sns.set()
            variables = list(x_kmeans)
            variables.remove('cluster')
            sns_matrix = sns.pairplot(x_kmeans, vars=variables, hue="cluster", plot_kws={"s": 25},
                                      diag_kind="hist")
            sns_matrix.fig.suptitle("Scatter matrix using {}".format(algorithm[0]))
            sns_matrix.fig.subplots_adjust(wspace=.03, hspace=.03)
            plt.show()
            print("")

            plt.show()
            if save_figs:
                sns_matrix.savefig("Imagenes/{}_sparse_{}.png".format(algorithm[0], cnt))
    return result


def test_params(data, params, models, names):
    test_algorithms = []
    if len(params) != len(models) and len(params) != len(names):
        exit(1)

    for model_params, model, name in zip(params, models, names):
        for param in model_params:
            model.set_params(**param)
            test_algorithms.append((name + str(param), sklearn.base.clone(model)))
    return exec_case(data, test_algorithms, plot_figs=False)


if __name__ == "__main__":
    threads = -1
    seed = 0
    censo = pd.read_csv('mujeres_fecundidad_INE_2018.csv')
    cnt = 1
    algoritmos = (
        ('K-Means', cluster.KMeans(init='k-means++', n_clusters=3, n_jobs=threads, random_state=seed)),
        ('Mean-Shift', cluster.MeanShift(bandwidth=0.5, n_jobs=threads)),
        ('DBSCAN', cluster.DBSCAN(eps=0.2, n_jobs=threads, )),
        ('Hierarchical-Clustering', cluster.AgglomerativeClustering(n_clusters=3)),
        ('BIRCH', cluster.Birch(threshold=0.3, n_clusters=3))
    )

    for col in censo:
        censo[col].fillna(censo[col].mean(), inplace=True)

    # Caso 1
    """
    print("----------Ejecutando caso 1------")
    subset = censo.loc[(censo['EDAD'] > 19) & (censo['EDAD'] <= 30) & (censo['NHIJOS'] > 0)]
    # seleccionar casos
    # seleccionar variables de interés para clustering
    usadas = ['INGRESOS', 'NTRABA', 'NDESEOHIJO', 'NHIJOS']
    X = subset[usadas]
    X_normal = X.apply(norm_to_zero_one)

    generate_table(exec_case(X_normal, algoritmos, plot_figs=True, save_figs=False))

    # Se prueban parametros de dos de los algoritmos

    generate_table(test_params(X_normal, [[{'n_clusters': 3}, {'n_clusters': 5}, {'n_clusters': 7}, {'n_clusters': 9}],
                                          [{'n_clusters': 3, 'threshold': 0.3}, {'n_clusters': 3, 'threshold': 0.2},
                                           {'n_clusters': 3, 'threshold': 0.1}, {'n_clusters': 5, 'threshold': 0.1}]],
                               [cluster.KMeans(init='k-means++', n_jobs=threads, random_state=seed), cluster.Birch()],
                               ['k-Means', 'Birch']))
    #"""
    """
    # Caso 2
    cnt = 2
    algoritmos = (
        ('K-Means', cluster.KMeans(init='k-means++', n_clusters=3, n_jobs=threads, random_state=seed)),
        ('Mean-Shift', cluster.MeanShift(bandwidth=0.3, n_jobs=threads)),
        ('DBSCAN', cluster.DBSCAN(eps=0.2, n_jobs=threads)),
        ('Hierarchical-Clustering', cluster.AgglomerativeClustering(n_clusters=3)),
        ('BIRCH', cluster.Birch(threshold=0.15))
    )
    subset = censo.loc[(censo['INTENTOEMB'] == 1)]
    # seleccionar casos
    # seleccionar variables de interés para clustering
    usadas = ['NDESEOHIJO', 'ESTUDIOSA', 'TEMPRELA', 'EDAD']
    X = subset[usadas]
    X_normal = X.apply(norm_to_zero_one)

    generate_table(exec_case(X_normal, algoritmos, plot_figs=True, save_figs=False))

    # Se prueban parametros de dos de los algoritmos

    generate_table(test_params(X_normal, [[{'n_clusters': 4}, {'n_clusters': 5}, {'n_clusters': 6}, {'n_clusters': 7}],
                                          [{'n_clusters': 3, 'threshold': 0.2}, {'n_clusters': 3, 'threshold': 0.25},
                                           {'n_clusters': 3, 'threshold': 0.1}, {'n_clusters': 5, 'threshold': 0.1},
                                           {'n_clusters': 5, 'threshold': 0.2}, {'n_clusters': 5, 'threshold': 0.25}

                                           ]],
                               [cluster.AgglomerativeClustering(), cluster.Birch()],
                               ['Hierarchical-Clustering', 'BIRCH']))
    """
    # Caso 3
    #"""
    cnt=3
    algoritmos = (
        ('K-Means', cluster.KMeans(init='k-means++', n_clusters=3, n_jobs=threads, random_state=seed)),
        ('Mean-Shift', cluster.MeanShift(bandwidth=0.2, n_jobs=threads)),
        ('DBSCAN', cluster.DBSCAN(eps=0.1, n_jobs=threads, )),
        ('Hierarchical-Clustering', cluster.AgglomerativeClustering(n_clusters=3)),
        ('BIRCH', cluster.Birch(threshold=0.1, n_clusters=4))
    )
    subset = censo.loc[(censo['EDAD'] > 25) or (censo)]
    # seleccionar casos
    # seleccionar variables de interés para clustering
    usadas = ['', '', '', '']
    X = subset[usadas]
    X_normal = X.apply(norm_to_zero_one)

    generate_table(exec_case(X_normal, algoritmos, plot_figs=True, save_figs=False))

    # Se prueban parametros de dos de los algoritmos
    
    generate_table(test_params(X_normal, [[{'n_clusters': 3}, {'n_clusters': 5}, {'n_clusters': 7}, {'n_clusters': 9}],
                                          [{'n_clusters': 3, 'threshold': 0.2}, {'n_clusters': 3, 'threshold': 0.25},
                                           {'n_clusters': 3, 'threshold': 0.1}, {'n_clusters': 5, 'threshold': 0.1}]],
                               [cluster.KMeans(init='k-means++', n_jobs=threads, random_state=seed), cluster.Birch()],
                               ['k-Means', 'Birch']))
    """
    #"""
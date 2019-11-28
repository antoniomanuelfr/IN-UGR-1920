# -*- coding: utf-8 -*-
"""
Autor:
    Jorge Casillas
Fecha:
    Noviembre/2019
Contenido:
    Ejemplo de uso de clustering en Python
    Inteligencia de Negocio
    Grado en Ingeniería Informática
    Universidad de Granada
"""

'''
Documentación sobre clustering en Python:
    http://scikit-learn.org/stable/modules/clustering.html
    http://www.learndatasci.com/k-means-clustering-algorithms-python-intro/
    http://hdbscan.readthedocs.io/en/latest/comparing_clustering_algorithms.html
    https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
    http://www.learndatasci.com/k-means-clustering-algorithms-python-intro/
'''

import time

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import sklearn.cluster as cluster
from sklearn import metrics
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from math import floor
import seaborn as sns

def norm_to_zero_one(df):
    
    return (df - df.min()) * 1.0 / (df.max() - df.min())
def generate_table (res):
    print("Algoritmo ; Tiempo; Calinski; Silhouette")
    for actual in res:
        print ("{}; {}s; {}; {} ".format(actual, res[actual][1],res[actual][2],res[actual][3]))

censo = pd.read_csv('mujeres_fecundidad_INE_2018.csv')

'''
for col in censo:
   missing_count = sum(pd.isnull(censo[col]))
   if missing_count > 0:
      print(col,missing_count)
#'''

#Se pueden reemplazar los valores desconocidos por un número
#censo = censo.replace(np.NaN,0)

#O imputar, por ejemplo con la> media      
for col in censo:
   censo[col].fillna(censo[col].mean(), inplace=True)
      
#seleccionar casos
subset = censo.loc[(censo['EDAD']>20) & (censo['EDAD']<=40)]

#seleccionar variables de interés para clustering
usadas = ['RELIGION', 'NHOGAR', 'NTRABA', 'TEMPRELA', 'NDESEOHIJO']
X = subset[usadas]

X_normal = X.apply(norm_to_zero_one)

print('----- Ejecutando k-Means',end='')
k_means = cluster.KMeans(init='k-means++', n_clusters=5, n_init=5)
ms = cluster.MeanShift()
algoritmos = (('kmeans',k_means), ('Mean Shift', ms))
result={'kmeans':[], 'Mean Shift': []}

for actual in algoritmos:

    t = time.time()

    cluster_predict = actual[1].fit_predict(X_normal)
    tiempo = time.time() - t

    result[actual[0]].append(cluster_predict)
    result[actual[0]].append(tiempo)

    metric_CH = metrics.calinski_harabaz_score(X_normal, cluster_predict)
    result[actual[0]].append(metric_CH)

    #el cálculo de Silhouette puede consumir mucha RAM. Si son muchos datos, digamos más de 10k, se puede seleccionar una muestra, p.ej., el 20%
    if len(X) > 10000:
       muestra_silhoutte = 0.2
    else:
       muestra_silhoutte = 1.0

    metric_SC = metrics.silhouette_score(X_normal, cluster_predict, metric='euclidean', sample_size=floor(muestra_silhoutte*len(X)), random_state=123456)
    result[actual[0]].append(metric_SC)


    #se convierte la asignación de clusters a DataFrame
    clusters = pd.DataFrame(cluster_predict,index=X.index,columns=['cluster'])

    print("Tamaño de cada cluster usando {} :".format(actual[0]))
    size=clusters['cluster'].value_counts()
    for num,i in size.iteritems():
       print('%s: %5d (%5.2f%%)' % (num,i,100*i/len(clusters)))

    centers = pd.DataFrame(k_means.cluster_centers_,columns=list(X))
    centers_desnormal = centers.copy()

    #se convierten los centros a los rangos originales antes de normalizar
    for var in list(centers):
        centers_desnormal[var] = X[var].min() + centers[var] * (X[var].max() - X[var].min())

    sns.heatmap(centers, cmap="YlGnBu", annot=centers_desnormal, fmt='.3f')

    '''
    print("---------- Preparando el scatter matrix...")
    #se añade la asignación de clusters como columna a X
    X_kmeans = pd.concat([X, clusters], axis=1)
    sns.set()
    variables = list(X_kmeans)
    variables.remove('cluster')
    sns_plot = sns.pairplot(X_kmeans, vars=variables, hue="cluster", palette='Paired', plot_kws={"s": 25}, diag_kind="hist") #en hue indicamos que la columna 'cluster' define los colores
    sns_plot.fig.subplots_adjust(wspace=.03, hspace=.03);
    sns_plot.savefig(actual[0] + ".png")
    print("")
    #'''
generate_table(result)
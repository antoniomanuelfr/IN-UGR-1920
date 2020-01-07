import pandas as pd
import numpy as np
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import preprocessing
import xgboost.sklearn as xgb
import lightgbm as lgb


data_path = "Entrega/Datos"
image_path = "Imagenes"
submission_path = "Submision"
seed = 10

np.random.seed(seed)


def validacion_cruzada(modelo, X, y, cv):
    y_test_all = []
    cnt = 0

    for train, test in cv.split(X, y):
        t = time.time()
        modelo = modelo.fit(X[train], y[train])
        tiempo = time.time() - t
        y_pred = modelo.predict(X[test])
        print("F1 score (val): {:.4f}, tiempo: {:6.2f} segundos".format(f1_score(y[test], y_pred, average='micro'),
                                                                        tiempo))
        y_test_all = np.concatenate([y_test_all, y[test]])
        cnt+=1

    print("")

    return modelo, y_test_all


if __name__ == "__main__":
    # Lectura de datos ya procesados

    X_train = pd.read_csv("{}/X_train_procesado.csv".format(data_path))
    y_train = pd.read_csv("{}/y_train_procesado.csv".format(data_path))
    X_test = pd.read_csv("{}/X_test_procesado.csv".format(data_path))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    '''
    print("------ AdaBoost...")
    clf = AdaBoostClassifier(random_state=seed)
    params_rf = {'n_estimators': [1000,110,1200,1300]}
    
    # '''
    """
    grid = GridSearchCV(clf, params_rf, cv=3, n_jobs=-1, scoring=make_scorer(f1_score, average='micro'), verbose=4)
    grid.fit(X_train.values, y_train.values.ravel())
    print("Los mejores parametros encontrados son: {}".format(grid.best_params_))
    #"""
    """
    print("------ XGBoost...")
    clf =xgb.XGBClassifier(nthread=8)
    params_rf = {'max_depth': [8, 10],'n_estimators': [100, 200, 300, 400] }
    #
    print ("Grid Search")
    grid = GridSearchCV(clf, params_rf, cv=3, n_jobs=-1, scoring=make_scorer(f1_score, average='micro'), verbose=4)
    grid.fit(X_train.values, y_train.values.ravel())
    print("Los mejores parametros encontrados son: {}".format(grid.best_params_))
    # Creo un modelo con los parametros anteriores
    print("Validacion cruzada")
    # clf = AdaBoostClassifier(random_state=seed)
    clf =xgb.XGBClassifier(nthread=8, **grid.best_params_)

    validacion_cruzada(clf, X=X_train.values, y=y_train.values.ravel(), cv=skf)
    clf = clf.fit(X_train.values, y_train.values.ravel())
    y_pred_tra = clf.predict(X_train.values)
    print("F1 score (tra): {:.4f}".format(f1_score(y_train.values, y_pred_tra, average='micro')))
    y_pred_tst = clf.predict(X_test.values)

    df_submission = pd.read_csv('{}/nepal_earthquake_submission_format.csv'.format(submission_path))
    df_submission['damage_grade'] = y_pred_tst
    df_submission.to_csv("{}/submission.csv".format(submission_path), index=False)
    """
        # '''
    print("------ Random Forest...")
    rf = RandomForestClassifier(random_state=seed, n_jobs=-1)
    params_rf = {'n_estimators': [20, 30, 50]}

    param_f1 = {'average': 'micro'}
    grid = GridSearchCV(rf, params_rf, cv=5, n_jobs=-1, scoring=make_scorer(f1_score, average='micro'), verbose = 4)
    grid.fit(X_train.values, y_train.values.ravel())
    #lgbm, y_test_lgbm = validacion_cruzada(lgbm, X_train.values, y_train.values.ravel(), skf)
    # '''

    # clf = xgbclf
    clf = grid.best_estimator_
    clf = clf.fit(X_train.values, y_train.values.ravel())
    y_pred_tra = clf.predict(X_train.values)
    print("F1 score (tra): {:.4f}".format(f1_score(y_train.values, y_pred_tra, average='micro')))
    y_pred_tst = clf.predict(X_test.values)

    df_submission = pd.read_csv('{}/nepal_earthquake_submission_format.csv'.format(submission_path))
    df_submission['damage_grade'] = y_pred_tst
    df_submission.to_csv("{}/submission.csv".format(submission_path), index=False)

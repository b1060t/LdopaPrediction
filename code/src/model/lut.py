from sklearn.svm import SVC
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost.sklearn import XGBClassifier

import sys
sys.path.append('..')
from feature import gen_pca, load_radiomics, load_volume

import scikitplot as skplt
import matplotlib.pyplot as plt

from sklearn import metrics

import numpy as np
import seaborn as sns

import functools

def class2_roc_auc_score(y_true, y_score):
    y_pred = y_score[:, 1]
    return metrics.roc_auc_score(y_true=y_true, y_score=y_pred)

def plot_r2(y_true, y_score):
    return sns.regplot(x=y_true, y=y_score)

def plot_confusion_matrix(y_true, y_score):
    mat = metrics.confusion_matrix(y_true, y_score)
    mat = mat.astype('float') / mat.sum(axis=1)[:, np.newaxis]
    return sns.heatmap(mat, square=True, annot=True, cmap=sns.color_palette("Blues", as_cmap=True))

Model_LUT = {
    'svc': SVC,
    'xgboost': XGBClassifier,
    'lasso': Lasso,
    'linear': LinearRegression,
    'logistic': LogisticRegression,
    'knn': KNeighborsClassifier,
    'nb': GaussianNB,
    'mlp': MLPClassifier
}

Metrics_LUT = {
    'AUC': class2_roc_auc_score,
    'AUC_multi': metrics.roc_auc_score,
    'r2': metrics.r2_score,
    'Accuracy': metrics.accuracy_score
}

Plot_LUT = {
    'plot_roc': skplt.metrics.plot_roc,
    'plot_confusion_matrix': plot_confusion_matrix,
    'plot_r2': plot_r2
}

Feature_LUT = {
    't1_pca': gen_pca,
    'gm_pca': gen_pca,
    'wm_pca': gen_pca,
    't1_radiomic': load_radiomics,
    't1_radiomic_full': load_radiomics,
    'gm_radiomic': load_radiomics,
    'tiv_gmv': load_volume
}
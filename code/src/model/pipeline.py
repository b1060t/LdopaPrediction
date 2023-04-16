import pandas as pd
import numpy as np
import os
import os.path
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
from sklearn.feature_selection import RFECV
from sklearn.linear_model import Lasso, LassoCV, LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import bootstrap
from mrmr import mrmr_classif, mrmr_regression
sys.path.append('..')
from src.utils.data import getPandas, getConfig, getDict
from src.model.lut import Model_LUT, Metrics_LUT, Feature_LUT, Plot_LUT
from src.model.stats import stats_analyze

def run(dataname, TASKS, FEATURES, log_func):
    data = getPandas(dataname)
    prefix = dataname.split('_')[0]
    task_config = getConfig('task')
    data_config = getConfig('data')
    img_config = getConfig('image')
    group = data_config['data_group']
    rst = {}
    
    for task_name in TASKS:
        log_func('Task: {}\n'.format(task_name))
        for feature_names in FEATURES:
            log_func('Feature: {}\n'.format(feature_names))
            
            task = task_config['task'][task_name]
            models = task['models']
            metric_list = task['metrics']
            
            x = data[group['demo'] + group['clinic']]
            y = data[[task['output']]]
            train_inds = data_config['indices'][prefix]['train']
            test_inds = data_config['indices'][prefix]['test']
            x_clinic_train = x.iloc[train_inds].reset_index(drop=True)
            x_clinic_test = x.iloc[test_inds].reset_index(drop=True)
            y_train = y.iloc[train_inds].reset_index(drop=True)
            y_test = y.iloc[test_inds].reset_index(drop=True)
            
            stats_analyze(x_clinic_train, x_clinic_test, y_train, y_test, data_config, log_func)
            
            x_img_train = pd.DataFrame(index=x_clinic_train.index).reset_index(drop=True)
            x_img_test = pd.DataFrame(index=x_clinic_test.index).reset_index(drop=True)
            
            for feature_name in feature_names:
                func = Feature_LUT[feature_name]
                params = img_config['task'][feature_name]['params']
                x_fe_train, x_fe_test = func(data, train_inds, test_inds, params)
                x_img_train = x_img_train.join(x_fe_train.reset_index(drop=True))
                x_img_test = x_img_test.join(x_fe_test.reset_index(drop=True))
                
            scaler = MinMaxScaler()
            x_img_train = pd.DataFrame(scaler.fit_transform(x_img_train, y_train), columns=x_img_train.columns)
            x_img_test = pd.DataFrame(scaler.transform(x_img_test), columns=x_img_test.columns)
            
            
            isContinuous = task['continuous']
            
            ##mRMR
            selected = mrmr_classif(X=x_img_train, y=y_train, K=50) if not isContinuous else mrmr_regression(X=x_img_train, y=y_train, K=50)
            log_func('Selected features: {}\n'.format(selected))
            ##LASSO
            la = LassoCV(cv=5, random_state=1, max_iter=10000)
            la.fit(x_img_train[selected], y_train)
            log_func('Selected alpha: {}\n'.format(la.alpha_))
            la.fit(x_img_train[selected], y_train)
            selected = np.array(selected)[np.abs(la.coef_)>0]
            log_func('Selected features: {}\n'.format(selected))
            ##RFE
            est = LogisticRegression(random_state=1) #l2
            selector = RFECV(est, min_features_to_select=1, cv=5, step=1)
            selector = selector.fit(X=x_img_train[selected], y=y_train)
            selected = np.array(selected)[selector.get_support()]
            log_func('Selected features: {}\n'.format(selected))
            
            # selected1 = ['rTHA_original_gldm_LargeDependenceHighGrayLevelEmphasis',
            #             'rTHA_original_gldm_SmallDependenceLowGrayLevelEmphasis',
            #             'rCAU_original_glcm_Idn',
            #             'cobra_wm_lInfPostCerebLIX',
            #             'cobra_gm_rFimbra',
            #             'cobra_wm_lAntCerebLIII']
            
            x_img_train = x_img_train[selected]
            x_img_test = x_img_test[selected]
            
            # Rearrange data
            # demo + clinic, demo + img, demo + clinic + img
            x_clinic_train = x_clinic_train.reset_index(drop=True)
            x_clinic_test = x_clinic_test.reset_index(drop=True)
            x_demo_train = x_clinic_train[group['demo']]
            x_demo_test = x_clinic_test[group['demo']]
            x_clinic_img_train = x_clinic_train.join(x_img_train)
            x_clinic_img_test = x_clinic_test.join(x_img_test)
            # No demo data in img df
            x_img_train = x_demo_train.join(x_img_train)
            x_img_test = x_demo_test.join(x_img_test)

            x_train_list = [x_clinic_train, x_img_train, x_clinic_img_train]
            x_test_list = [x_clinic_test, x_img_test, x_clinic_img_test]
            info_list = ['Demo + Clinic:', 'Demo + Img:', 'Demo + Clinic + Img:']
            
            x_train_list = [x_clinic_train, x_clinic_img_train]
            x_test_list = [x_clinic_test, x_clinic_img_test]
            info_list = ['Demo + Clinic:', 'Demo + Clinic + Img:']
            
            # x_train_list = [x_clinic_img_train]
            # x_test_list = [x_clinic_img_test]
            # info_list = ['Demo + Clinic + Img:']
            
            #Main loop
            for i in range(len(info_list)):
                x_train = x_train_list[i]
                x_test = x_test_list[i]
                
                scaler = MinMaxScaler()
                
                x_train = pd.DataFrame(scaler.fit_transform(x_train, y_train), columns=x_train.columns)
                x_test = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns)
                
                log_func(info_list[i] + '\n')
                
                for model in models:
                    name = model['name']
                    parameters = model['params']
                    model = Model_LUT[name]
                    
                    cv = GridSearchCV(
                        model(),
                        parameters,
                        n_jobs=5,
                        # StratifiedGroupKFold?
                        cv=(StratifiedKFold(n_splits=5, shuffle=True, random_state=2) if task['stratify'] else KFold(n_splits=5, shuffle=True, random_state=1)),
                        scoring=task['gridsearch_params']['scoring']
                    )

                    cv.fit(x_train, y_train.values.ravel())
                    log_func('Best params: {}\n'.format(cv.best_params_))
                    model_instance = model(**cv.best_params_)
                    model_instance.fit(x_train, y_train.values.ravel())
                    
                    log_func('Model: {}\n'.format(name))
                    
                    for metric in metric_list:
                        metric_func = Metrics_LUT[metric[0]]
                        # predict_proba True
                        # predic        False
                        pred_func = model_instance.predict_proba if metric[1] else model_instance.predict
                        train_pred = pred_func(x_train)
                        test_pred = pred_func(x_test)
                        
                        # CI
                        from sklearn import metrics
                        y_train_np = y_train[task['output']].to_numpy()
                        y_test_np = y_test[task['output']].to_numpy()
                        ci_train = bootstrap((y_train_np, train_pred[:, 1]), statistic=metrics.roc_auc_score, n_resamples=1000, paired=True, random_state=1)
                        ci_test = bootstrap((y_test_np, test_pred[:, 1]), statistic=metrics.roc_auc_score, n_resamples=1000, paired=True, random_state=1)
                        
                        log_func('{} train {} ci {}\n'.format(metric[0],
                                                            metric_func(y_train_np, train_pred),
                                                            ci_train.confidence_interval
                                                            ))
                        log_func('{} test {} ci {}\n'.format(metric[0],
                                                            metric_func(y_test_np, test_pred),
                                                            ci_test.confidence_interval
                                                            ))
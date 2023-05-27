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
from src.model.lut import Model_LUT, Metrics_LUT, Plot_LUT
from src.model.feature import Feature_LUT
from src.model.stats import stats_analyze

def run(dataname, TASKS, FEATURES, log_func=print, plot_flag=True, feature_selection=True, cal_baselines=False):
    data = getPandas(dataname)
    prefix = dataname.split('_')[0]
    task_config = getConfig('task')
    data_config = getConfig('data')
    img_config = getConfig('image')
    group = data_config['data_group']
    rst = {}
    rst['img'] = {}
    rst['baseline'] = {}
    
    for task_name in TASKS:
        log_func('Task: {}\n'.format(task_name))
        for feature_names in FEATURES:
            log_func('Feature: {}\n'.format(feature_names))
            
            task = task_config['task'][task_name]
            random_states = task['random_state']
            models = task['models']
            metric_list = task['metrics']

            x = data[group['demo'] + group['clinic']]
            y = data[[task['output']]]
            img_param = data[group['img']]
            train_inds = data_config['indices'][prefix]['train']
            test_inds = data_config['indices'][prefix]['test']
            x_clinic_train = x.iloc[train_inds].reset_index(drop=True)
            x_clinic_test = x.iloc[test_inds].reset_index(drop=True)
            x_param_train = img_param.iloc[train_inds].reset_index(drop=True)
            x_param_test = img_param.iloc[test_inds].reset_index(drop=True)
            y_train = y.iloc[train_inds].reset_index(drop=True)
            y_test = y.iloc[test_inds].reset_index(drop=True)
            
            stats_analyze(x_clinic_train, x_clinic_test, y_train, y_test, data_config, log_func)

            for random_state in random_states:
            
                x_img_train = pd.DataFrame(index=x_clinic_train.index).reset_index(drop=True)
                x_img_test = pd.DataFrame(index=x_clinic_test.index).reset_index(drop=True)
            
                for feature_name in feature_names:
                    func = Feature_LUT[feature_name]
                    params = img_config['task'][feature_name]['params']
                    x_fe_train, x_fe_test = func(data, train_inds, test_inds, params)
                    x_img_train = x_img_train.join(x_fe_train.reset_index(drop=True))
                    x_img_test = x_img_test.join(x_fe_test.reset_index(drop=True))
                
                #scaler = MinMaxScaler()
                #x_img_train = pd.DataFrame(scaler.fit_transform(x_img_train, y_train), columns=x_img_train.columns)
                #x_img_test = pd.DataFrame(scaler.transform(x_img_test), columns=x_img_test.columns)
            
                isContinuous = task['continuous']
                if feature_selection:
                    ##mRMR
                    selected = mrmr_classif(X=x_img_train, y=y_train, K=50) if not isContinuous else mrmr_regression(X=x_img_train, y=y_train, K=50)
                    log_func('mRMR Selected features: {}\n'.format(selected))
                    ##LASSO
                    la = LassoCV(cv=5, random_state=random_state, max_iter=10000)
                    la.fit(x_img_train[selected], y_train)
                    log_func('LASSO Selected alpha: {}\n'.format(la.alpha_))
                    la.fit(x_img_train[selected], y_train)
                    if plot_flag:
                        plt.semilogx(la.alphas_, la.mse_path_, ':')
                        plt.plot(
                            la.alphas_ ,
                            la.mse_path_.mean(axis=-1),
                            "k",
                            label="Average across the folds",
                            linewidth=2,
                        )
                        plt.axvline(
                            la.alpha_, linestyle="--", color="k", label="alpha: CV estimate"
                        )
                        plt.legend()
                        plt.ylabel('MSE')
                        plt.xlabel('alpha')
                        plt.show()
                        la = Lasso(alpha=la.alpha_)
                        la.fit(x_img_train[selected], y_train)
                        selected = np.array(selected)[np.abs(la.coef_)>0]
                        coef = np.array(la.coef_)[np.abs(la.coef_)>0]
                        sort_idx = coef.argsort()
                        plt.barh(selected[sort_idx], coef[sort_idx])
                        plt.xlabel('Importance')
                        plt.show()
                    else:
                        selected = np.array(selected)[np.abs(la.coef_)>0]
                    log_func('LASSO Selected features: {}\n'.format(selected))
                    if len(selected) > 2:
                        ##RFE
                        est = LogisticRegression(random_state=random_state) #l2
                        selector = RFECV(est, min_features_to_select=1, cv=5, step=1)
                        selector = selector.fit(X=x_img_train[selected], y=y_train)
                        selected = np.array(selected)[selector.get_support()]
                        if plot_flag:
                            n_scores = len(selector.cv_results_["mean_test_score"])
                            plt.errorbar(
                                range(1, n_scores+1),
                                selector.cv_results_["mean_test_score"],
                                yerr=selector.cv_results_["std_test_score"],
                            )
                            plt.xticks(range(1,n_scores+1))
                            plt.xlabel("Number of features selected")
                            plt.ylabel("Mean test accuracy")
                            plt.show()
                            coef = selector.estimator_.coef_[0]
                            sort_idx = coef.argsort()
                            plt.barh(selected[sort_idx], coef[sort_idx])
                            plt.xlabel('Importance')
                            plt.show()
                        log_func('RFE Selected features: {}\n'.format(selected))
                    selected = [
                        'rSN_original_glcm_ClusterProminence',
                        'rCAU_original_gldm_LargeDependenceHighGrayLevelEmphasis',
                        'rTHA_original_glszm_LargeAreaHighGrayLevelEmphasis'
                    ]
                
                    x_img_train = x_img_train[selected]
                    x_img_test = x_img_test[selected]
            
                #x_img_train = x_img_train.join(x_param_train)
                #x_img_test = x_img_test.join(x_param_test)
            
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
            
                if not cal_baselines:
                    x_train_list = [x_clinic_img_train]
                    x_test_list = [x_clinic_img_test]
                    info_list = ['Demo + Clinic + Img:']
            
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
                        parameters['random_state'] = [random_state]
                        model = Model_LUT[name]

                        if name not in rst['baseline']:
                            rst['baseline'][name] = []
                        if name not in rst['img']:
                            rst['img'][name] = []
                    
                        cv = GridSearchCV(
                            model(),
                            parameters,
                            n_jobs=5,
                            # StratifiedGroupKFold?
                            cv=(StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state) if task['stratify'] else KFold(n_splits=5, shuffle=True, random_state=random_state)),
                            scoring=task['gridsearch_params']['scoring']
                        )

                        cv.fit(x_train, y_train.values.ravel())
                        #log_func('Best params: {}\n'.format(cv.best_params_))
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
                            #ci_train = bootstrap((y_train_np, train_pred[:, 1]), statistic=metrics.roc_auc_score, n_resamples=1000, paired=True, random_state=random_state)
                            #ci_test = bootstrap((y_test_np, test_pred[:, 1]), statistic=metrics.roc_auc_score, n_resamples=1000, paired=True, random_state=random_state)
                        
                            log_func('{} train {}\n'.format(metric[0],
                                                                metric_func(y_train_np, train_pred)
                                                                ))
                            #log_func('{}\n'.format(ci_train.confidence_interval))
                            log_func('{} test {}\n'.format(metric[0],
                                                                metric_func(y_test_np, test_pred)
                                                                ))
                            #log_func('{}\n'.format(ci_test.confidence_interval))

                            if metric[0] == 'AUC':
                                if len(info_list) == 1:
                                    rst['img'][name].append(metric_func(y_test_np, test_pred))
                                elif len(info_list) == 2:
                                    if i == 0:
                                        rst['baseline'][name].append(metric_func(y_test_np, test_pred))
                                    else:
                                        rst['img'][name].append(metric_func(y_test_np, test_pred))
                                else:
                                    if i == 0:
                                        rst['baseline'][name].append(metric_func(y_test_np, test_pred))
                                    elif i == 2:
                                        rst['img'][name].append(metric_func(y_test_np, test_pred))
                            
                        plot_list = task['plot']
                        if plot_flag:
                            for plot in plot_list:
                                plot_func = Plot_LUT[plot[0]]
                                pred_func = model_instance.predict_proba if plot[1] else model_instance.predict
                                train_pred = pred_func(x_train)
                                test_pred = pred_func(x_test)
                                plot_func(y_test[task['output']].to_numpy(), test_pred)
                                plt.show()
    return rst['baseline'], rst['img']
import pandas as pd
from scipy.stats import ttest_ind, chi2_contingency, normaltest, ranksums

def stats_analyze(x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame, data_config):
    # class_tags
    for tag in data_config['class_tags']['x']:
        if tag in x_train.columns:
            train = x_train[[tag]].value_counts()
            test = x_test[[tag]].value_counts()
            _, p, _, _ = chi2_contingency([train, test])
            print('{} chi2 p: {}'.format(tag, p))
    for tag in data_config['class_tags']['y']:
        if tag in y_train.columns:
            train = y_train[[tag]].value_counts()
            test = y_test[[tag]].value_counts()
            _, p, _, _ = chi2_contingency([train, test])
            print('{} chi2 p: {}'.format(tag, p))
    for tag in data_config['cont_tags']['x']:
        if tag in x_train.columns:
            train = x_train[[tag]]
            test = x_test[[tag]]
            p_train = normaltest(train).pvalue
            p_test = normaltest(test).pvalue
            print('{} Normaltest p_train: {}, p_test: {}'.format(tag, p_train, p_test))
            if (p_train >= 0.05) & (p_test >= 0.05):
                print('{} t-test p: {}'.format(tag, ttest_ind(train, test)))
            else:
                print('{} ranksums p: {}'.format(tag, ranksums(train, test)))
    for tag in data_config['cont_tags']['y']:
        if tag in y_train.columns:
            train = y_train[[tag]]
            test = y_test[[tag]]
            p_train = normaltest(train).pvalue
            p_test = normaltest(test).pvalue
            print('{} Normaltest p_train: {}, p_test: {}'.format(tag, p_train, p_test))
            if (p_train >= 0.05) & (p_test >= 0.05):
                print('{} t-test p: {}'.format(tag, ttest_ind(train, test)))
            else:
                print('{} ranksums p: {}'.format(tag, ranksums(train, test)))
    print()
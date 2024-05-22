#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: naresh
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from scipy.stats import zscore
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor

def impute_nulls(train, test=pd.DataFrame(), num_impute_type='median'):
    """
    Function to find the nulls/NaNs and impute values with corresponding impute methods.
    'train' dataset is necessary parameter and rest are optional parameters
        Parameters:
            train (dataframe): Dataframe dataset
            test (dataframe): Dataframe dataset
            num_impute_type (string): 'mean' or 'median' impute methods for numerical columns
            for categorical columns, 'mode' is used

        Returns:
            train (dataframe): Imputed dataframe
            test (dataframe): Imputed dataframe
    """
    train = train
    test = test
    num_impute_type = num_impute_type
    print('\n')
    print('Nulls Treatment Started\n')
    print('Numerical data impute type:', num_impute_type)
    print('Categorical data impute type: mode')
    
    if train.shape[0] == 0 and train.shape[0] == 0:
        print("Given train and test datasets are empty")
        return train, test
    if train.shape[0] > 0:
        print('Impute percentage of each fetaure:')
        for col in train.columns:
            if train[col].dtype.name in ['int64','float64','object','category','bool']:
                if train[col].dtype.name in ['int64','float64']:
                    if num_impute_type == 'mean':
                        impute_value = train[col].mean()
                    else:
                        impute_value = train[col].median()
                else:
                    impute_value = train[col].mode()[0]
                idx = train.index[train[col].isnull()].tolist()
                idx.extend(train.index[train[col].isna()].tolist())
                idx.extend(train.index[train[col] == ''].tolist())
                idx = list(set(idx))
                tmp = train.filter(items=idx, axis=0)
                tmp[col] = impute_value
                train.update(tmp)
                if test.shape[0] > 0 and set(train.columns)-set(test.columns) == set():
                    idx = test.index[test[col].isnull()].tolist()
                    idx.extend(test.index[test[col].isna()].tolist())
                    idx.extend(test.index[test[col] == ''].tolist())
                    idx = list(set(idx))
                    tmp2 = test.filter(items=idx, axis=0)
                    tmp2[col] = impute_value
                    test.update(tmp2)
                    if tmp.shape[0] > 0 and test.shape[0] > 0:
                        print(f'{col:40} : train - {str(np.round(tmp.shape[0]*100/train.shape[0],2))+"%":10} test - {str(np.round(tmp2.shape[0]*100/test.shape[0],2))+"%":10}')    
                if tmp.shape[0] > 0 and test.shape[0] == 0:
                    print(f'{col:40} : train - {str(np.round(tmp.shape[0]*100/train.shape[0],2))+"%":10}')    
    print('\nNulls Treatment Completed')
    return train, test


def control_outliers(data, method='iqr', num_impute_type='median', impute=True, log_transform=False,
                     std=3,percentiles=[0.01, 0.01]):
    """
    Function to find the outliers and impute the values with corresponding impute methods.
    'data' is necessary parameter and 'rest are optional parameters
        Parameters:
            data (dataframe): Dataframe dataset
            method (string): Outlier filter methods - 'percentile' or iqr' or 'zscore'
            num_impute_type (string): 'mean' or 'median' or 'mode' or 'neighbour' impute value for
                                       numerical columns
            impute (string): True for imputing values and False for imputing NaN
            log_transform (string): True or False. Transform the data into log values to reduce the
                                    skewness(not always)
            std (int): number of standard deviations
            percentiles (list): list of bottom percentile and top percentile to filter the outliers

        Returns:
            data (dataframe): Imputed dataframe
    """
    data = data
    method = method
    keep = impute
    lower_percentile = percentiles[0]
    upper_percentile = 1-percentiles[1]
    std_value = std
    log_transform = log_transform
    
    print('\n')
    print('Outlier Treatment Started\n' )
    print('Log Transformation:', log_transform)
    if method == 'percentile':
        print('Outlier Selection Method:', method)
        print('Lower bound outliers percent:', percentiles[0]*100)
        print('Upper bound outliers percent:', percentiles[1]*100)
    elif method == 'iqr':
        print('Outlier Selection Method:', method)
    elif method == 'zscore':
        print('Outlier Selection Method:', method)
        print('Number of standard deviations: ', std_value)
    else:
        print('Outlier Selection Method:', method)
        
    if keep == True:
        print('Numerical data impute type:', num_impute_type)
        print('Categorical data impute type: mode')
    else:
        print('Outliers are replaced with None')
    
    for col in data.columns:
        if data[col].dtype.name in ['int64','float64']:
            if log_transform is True:
                data['log_' + col] = np.log(data[col])
                data.drop(columns=[col], inplace=True)
                col = 'log_' + col
            if method == 'percentile':
                lower_percentile_value = data.quantile(lower_percentile, interpolation='higher')
                upper_percentile_value = data.quantile(upper_percentile, interpolation='lower')
                if keep is True:
                    data.loc[data[col] < lower_percentile_value, col] = lower_percentile_value
                    data.loc[data[col] > upper_percentile_value, col] = upper_percentile_value
                else:
                    data.loc[data[col] < lower_percentile_value, col] = np.nan
                    data.loc[data[col] > upper_percentile_value, col] = np.nan
            elif method == 'iqr':
                q_1 = data[col].quantile(0.25, interpolation='higher')
                q_3 = data[col].quantile(0.75, interpolation='lower')
                iqr = q_3-q_1
                lower_limit = q_1 - 1.5*iqr
                upper_limit = q_3 + 1.5*iqr
                idx = data.index[~data[col].between(lower_limit, upper_limit, inclusive='both')].tolist()
                outlier_data = data.filter(items=idx, axis=0)
                standard_data = data.drop(index=idx, axis=0)
            elif method == 'zscore':
                lower_limit = -std_value
                upper_limit = std_value
                new_col = 'zscore' + col
                tmp = pd.DataFrame(columns=[new_col])
                tmp[new_col] = zscore(data[col])
                idx = tmp.index[~tmp[col].between(lower_limit, upper_limit, inclusive='both')].tolist()
                outlier_data = data.filter(items=idx, axis=0)
                standard_data = data.drop(index=idx, axis=0)
            else:
                raise ValueError('Please input correct method. Allowed methods are iqr, percentile and zscore')
                break
            if keep is True:
                if num_impute_type == 'median':
                    outlier_data[col] = standard_data[col].median()
                elif num_impute_type == 'mean':
                    outlier_data[col] = standard_data[col].mean()
                elif num_impute_type == 'mode':
                    outlier_data[col] = standard_data[col].mode()[0]
                elif num_impute_type == 'neighbour':
                    outlier_data.loc[outlier_data[col] < lower_limit, col] = standard_data.sort_values(by=[col], ascending=True)[col][0]
                    outlier_data.loc[outlier_data[col] > upper_limit, col] = standard_data.sort_values(by=[col], ascending=False)[col][0]
                data.update(outlier_data)
            elif keep is False:
                outlier_data[col] = np.nan
                data.update(outlier_data)
    print('\nOutlier Treatment Completed' )
    return data


def feature_encoding(data=pd.DataFrame(), encode_columns=[], encode_type='onehot', max_unique_values=20):
    """
    Function to encode the categorical variables.
    'data' is necessary parameter and 'encode_columns' & 'encode_type' are optional parameters
        Parameters:
            data (dataframe): Dataframe dataset
            encode_columns (list): List of columns that require encoding
            encode_type (string): 'onehot' or 'label' encoding methiods

        Returns:
            data (dataframe): Transformed dataframe
    """
    data = data
    encode_columns = encode_columns
    encode_type = encode_type
    max_unique_values = max_unique_values
    print('\n')
    print('Feature Encoding Started\n')
    print('Feature encoding type:', encode_type)
    print('Encoded the features with unique values not greater than', max_unique_values)        
    if data.shape[0] > 0:
        if len(encode_columns) == 0:
            cat_columns = [col for col in data.columns if data[col].dtype.name in ['object','category','bool']]
        else:
            cat_columns = encode_columns
        cat_columns = [col for col in cat_columns if data[col].agg(['nunique'])[0] <= max_unique_values]
        rest_columns = list(set(data.columns)-set(cat_columns))
        if encode_type == 'onehot':
            cat_data = pd.get_dummies(data[cat_columns])
            if len(rest_columns) > 0:
                rest_data = data[rest_columns]
                data = pd.concat([rest_data, cat_data], axis=1)
            else:
                data = cat_data
        else:
            data_tmp = pd.DataFrame(columns=cat_columns)
            for col in cat_columns:
                data_tmp[col] = data[col].astype('category').cat.codes

            if len(rest_columns) > 0:
                rest_data = data[rest_columns]
                data = pd.concat([rest_data, data_tmp], axis=1)
            else:
                data = data_tmp
    else:
        raise TypeError('No data input or input data has zero records')
    print('\nFeature Encoding Completed' )
    return data

def classification_models(x_train, y_train, params_log_reg={}, params_svc={'kernel':'linear'},
                          params_dtc={}, params_rfc={}, params_xgbc={}, models=[]):
    """
    Function to train the linear, logistic, decision trees.
    'train_data' is necessary parameter and remaining are optional parameters
        Parameters:
            x_train (dataframe): Dataframe dataset
            y_train (dataframe): Dataframe dataset
            params_log_reg (dict): logistic regression parameters
            params_dtc (dict): decision tree parameters
            params_svc (dict): SVC parameters
            params_rfc (dict): random forest classifier parameters
            params_xgbc (dict): xboost classifier parameters
            models (list): ['log_reg','svc','dtc','rfc','xgbc']

        Returns:
            log_reg (object): trained model output
            svc (object): trained model output
            dtc (object): trained model output
            rfc (object): trained model output
            xgbc (object): trained model output
    """
    params_log_reg = params_log_reg
    params_svc = params_svc
    params_dtc = params_dtc
    params_rfc = params_rfc
    params_xgbc = params_xgbc
    models = models
    log_reg = ''
    svc = ''
    dtc = ''
    rfc = ''
    xgbc = ''
    if models == [] or 'log_reg' in models:
        if params_log_reg == {}:
            log_reg = LogisticRegression().fit(x_train, y_train)
        else:
            log_reg = LogisticRegression(params_log_reg).fit(x_train, y_train)
        print('\n1: Executed Logistic Regression\n', log_reg.get_params())
    if models == [] or 'svc' in models:
        if params_svc == {}:
            svc = SVC().fit(x_train, y_train)
        else:
            svc = SVC(params_svc).fit(x_train, y_train)
        print('\n2: Executed SVC Linear\n', svc.get_params())
    if models == [] or 'dtc' in models:
        if params_dtc == {}:
            dtc = DecisionTreeClassifier().fit(x_train, y_train)
        else:
            dtc = DecisionTreeClassifier(params_dtc).fit(x_train, y_train)
        print('\n3: Executed Decision Tree Classifier\n', dtc.get_params())
    if models == [] or 'rfc' in models:
        if params_rfc == {}:
            rfc = RandomForestClassifier().fit(x_train, y_train)
        else:
            rfc = RandomForestClassifier(params_rfc).fit(x_train, y_train)
        print('\n4: Executed Random Forest Classifier\n', rfc.get_params())
    if models == [] or 'xgbc' in models:
        if params_xgbc == {}:
            xgbc = XGBClassifier().fit(x_train, y_train)
        else:
            xgbc = XGBClassifier(params_xgbc).fit(x_train, y_train)
        print('\n5: Executed XGBoost Classifier\n', xgbc.get_params())
    return log_reg, svc, dtc, rfc, xgbc


def feature_selection_vif(data, max_vif: float=10.0, top_features: int=0):
    """
    Function to select features based on the top features of the VIF scores.
    'data' is necessary parameter and remaining are optional parameters.
        Parameters:
            data (dataframe): Dataframe dataset
            max_vif (float): maximum VIF value (1 to 5 in general) allowed to select feature
            top_features (int): Number of top features to be selected

        Returns:
            vif (dataframe): Dataframe with feature and it's vif score
    """
    data=data
    max_vif = max_vif
    top_features = top_features
    print('\nVIF Based Feature Selection Started' )
    vif = pd.DataFrame()
    vif["feature"] = data.columns
    vif["vif"] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]
    vif.sort_values(by=['vif'], inplace=True)
    vif.reset_index(drop=True, inplace=True)
    if top_features > 0:
        print('The number of top features selected are', top_features)
        vif = vif[:top_features]
    else:
        print('The features are selected with VIF value not greater than', max_vif)
        vif = vif[vif['vif'].between(1, max_vif, inclusive='both')].reset_index(drop=True)
    print('\nVIF Based Feature Selection Completed' )
    return vif


def feature_selection_iv(model, min_threshold: float=0.0, top_features: int=0):
    """
    Function to select features based on the top features of the model.
    'model' is necessary parameter and remaining are optional parameters.
        Parameters:
            model (object): The trained classifier
            min_threshold (float): minimum threshold value (0 to 1) allowed to select feature
            top_features (int): Number of top features to be selected

        Returns:
            df (dataframe): Dataframe with feature and it's importance score
    """
    model=model
    min_threshold = min_threshold
    top_features = top_features
    print('\nIV Based Feature Selection Started' )
    df_imp = pd.DataFrame(zip(model.feature_names_in_, model.feature_importances_),columns=['feature','value'])
    df_imp.sort_values(by=['value'], inplace=True, ascending=False)
    df_imp.reset_index(drop=True, inplace=True)
    if top_features > 0:
        print('The number of top features selected are', top_features)
        df_imp = df_imp[:top_features]
    else:
        print('The features are selected with importance value not less than', min_threshold)
        df_imp = df_imp[df_imp['value']>=min_threshold]
    print('\nIV Based Feature Selection Completed' )
    return df_imp

# # Testing
# alloy_persons_data_path = '../data/alloy_persons_data.pkl'
# alloy_persons_data = pd.read_pickle(alloy_persons_data_path)

# x_train = alloy_persons_data.drop(columns=['fraud'])
# y_train = alloy_persons_data['fraud']

# output1, output2 = impute_nulls(train=x_train, test=x_train)

# df_outlier_treated = control_outliers(output1, method='iqr')
# df_feature_encoding = feature_encoding(data=df_outlier_treated)

# df_feature_encoding = df_feature_encoding.select_dtypes(exclude=['object'])
# x_train = df_feature_encoding
# y_train = y_train
# models = classification_models(x_train, y_train, models=['xgbc'])

# df_vif = feature_selection_vif(x_train)
# df_vi = feature_selection_iv(model=models[-1])

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 14:41:02 2020

@author: zhiyulin
"""

import pandas as pd
import numpy as np
import datetime

from matplotlib import pyplot as plt
from functools import reduce
from sklearn import preprocessing
from sklearn.pipeline import FeatureUnion, Pipeline 

### Simple formats for the given csv file
class simpleFormating():
    def __init__(self, path_to_analysis_dataset="Analysis_dataset.csv"):
        self.path_to_analysis_dataset = path_to_analysis_dataset
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        ## Drop unnamed column
        analysis_df = pd.read_csv(self.path_to_analysis_dataset).drop('Unnamed: 0', axis = 1)
        
        ## Convert signup and purchase times to pandas datetime
        analysis_df.signup_time = pd.to_datetime(analysis_df.signup_time, format = '%m/%d/%Y %H:%M')
        analysis_df.purchase_time = pd.to_datetime(analysis_df.purchase_time, format = '%m/%d/%Y %H:%M')
        
        ## Fill missing values with NA
        analysis_df = analysis_df.fillna('NA')
        return analysis_df
    
    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)

### Calculate ratio of fraudulent transaction by each categorical variable
class calculateRatioFraud():
    def __init__(self, sel_var=None):
        self.sel_var = sel_var

    def fit(self,X,y=None):
        return self 

    def transform(self,X):
        # copy the original df to tmp
        tmp = X.copy()
        # group by the variable of interest (country) and class
        tmp = tmp.groupby([self.sel_var, 'class']).user_id.nunique()\
        .unstack(level = 1)\
        .reset_index()\
        .rename(columns = {0:'Not Fraud', 1: 'Fraud'}).fillna(0.0)     
        # create two new variables in tmp df
        tmp['ratio_fraud_' + self.sel_var] = tmp['Fraud']/(tmp['Fraud'] + tmp['Not Fraud'])
        tmp['num_trans_' + self.sel_var] = tmp['Fraud'] + tmp['Not Fraud']        
        return X[['user_id', self.sel_var]]\
            .merge(tmp[[self.sel_var, 'ratio_fraud_' + self.sel_var, 'num_trans_' + self.sel_var]], on = self.sel_var)
        
    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)


### Calculate time between sign up and purchase
class calculateTimeLatency():
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        X['purchase_time'] = pd.to_datetime(X['purchase_time'])
        X['signup_time'] = pd.to_datetime(X['signup_time'])
        X['time_latency'] = (X['purchase_time'] - X['signup_time']).dt.total_seconds()/60/60
        return X[['user_id','time_latency']]
    
    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)
    
### Select features
class subsetFeatures():
    def __init__(self, cols = ['user_id', 'purchase_value', 'class']):
        self.cols = cols
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[self.cols]
    
    def fit_transform(self, X, y=None):
        return self.fit(X,y).transform(X)

### Encode multiple columns at once
class MultiColumnLabelEncoder:
    
    def __init__(self, columns = None):
        self.columns = columns 
        
    def fit(self,X,y=None):
        return self 

    def transform(self,X):
        '''
        Transform specified columns, if no specification, transform all columns
        '''
        output = X.copy()
        #output = output.apply(lambda col: preprocessing.LabelEncoder().fit_transform(col.astype(str)), axis=0, result_type='expand')
        if self.columns is not None:
            for col in self.columns:
                output[:,col] = preprocessing.LabelEncoder().fit_transform(output[:,col])
        else:
            for col in range(output.shape[1]):
                output[:,col] = preprocessing.LabelEncoder().fit_transform(output[:,col])
        
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)

### Delete certain features
class customSelector():
    
    def __init__(self, colnum):
        self.colnum = colnum
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return np.delete(X, self.colnum, axis=1)
    
    def fit_transform(self, X,y=None):
        return self.fit(X,y).transform(X)


'''
### Concatenate to final data
class concatToFinalFeatures():
    def __init__(self, var_list):
        self.var_list = var_list
    
    def fit(self, X1, X2, y=None):
        return self
    
    def transform(self,X1,X2):
        output = pd.concat([X1.drop(self.var_list, axis = 1), 
                            X2], axis = 1).set_index(['user_id', 'device_id'])
        return output
    
    def fit_transform(self,X1,X2,y=None):
        return self.fit(X1,X2,y).transform(X1,X2)
'''


### Pipeline preprocessing
feature_names = ['user_id','sex','ratio_fraud_sex','num_trans_sex',
                       'device_id','ratio_fraud_device_id','num_trans_device_id',
                       'age','ratio_fraud_age', 'num_trans_age',
                       'country','ratio_fraud_country','num_trans_country',
                       'browser','ratio_fraud_browser','num_trans_browser',
                       'source', 'ratio_fraud_source','num_trans_source',
                       'time_latency','purchase_value','class']

PL = Pipeline([
    ('simple_formating',simpleFormating()),
    ('features1',FeatureUnion([
        ('fraud_by_sex',calculateRatioFraud('sex')),
        ('fraud_by_device_id',calculateRatioFraud('device_id')),
        ('fraud_by_age',calculateRatioFraud('age')),
        ('fraud_by_country',calculateRatioFraud('country')),
        ('fraud_by_browser',calculateRatioFraud('browser')),
        ('fraud_by_source',calculateRatioFraud('source')),
        ('time_latency', calculateTimeLatency()),
        ('subsets', subsetFeatures())
    ])),
    ('select_features',customSelector([4,8,12,16,20,24,26])),   # all columns before encoding
    ('encode', MultiColumnLabelEncoder([1,10,13,16]))           # 'country', 'sex', 'browser', 'source'
    #('to_feature_df', concatToFinalFeatures([1,10,13,16]))
])



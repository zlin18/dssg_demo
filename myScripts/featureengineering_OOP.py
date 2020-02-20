import pandas as pd
import numpy as np
from datetime import datetime

from matplotlib import pyplot as plt
from functools import reduce
from sklearn import preprocessing

### Encode multiple columns at once
class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        output = output.apply(lambda col: preprocessing.LabelEncoder().fit_transform(col.astype(str)), axis=0, result_type='expand')
        if self.columns is not None:
            for col in self.columns:
                output[col] = preprocessing.LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = preprocessing.LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)
    
### Calculate ratio of fraudulent transaction by each categorical variable
class calculateRatioFraud():
    
    def __init__(self, df, sel_var):
        self.df = df
        self.sel_var = sel_var

    def fit(self):
        return self 

    def transform(self):
        # copy the original df to tmp
        tmp = self.df.copy()
        # group by the variable of interest (country) and class
        tmp = tmp.groupby([self.sel_var, 'class']).user_id.nunique()\
        .unstack(level = 1)\
        .reset_index()\
        .rename(columns = {0:'Not Fraud', 1: 'Fraud'}).fillna(0.0)     
        # create two new variables in tmp df
        tmp['ratio_fraud_' + self.sel_var] = tmp['Fraud']/(tmp['Fraud'] + tmp['Not Fraud'])
        tmp['num_trans_' + self.sel_var] = tmp['Fraud'] + tmp['Not Fraud']        
        return self.df[['user_id', self.sel_var]]\
            .merge(tmp[[self.sel_var, 'ratio_fraud_' + self.sel_var, 'num_trans_' + self.sel_var]], on = self.sel_var)
        
    def fit_transform(self):
        return self.fit().transform()

### Calculate time between sign up and purchase
class calculateTimeLatency():
    def __init__(self, df):
        self.df = df
        
    def fit(self):
        return self
    
    def transform(self):
        self.df['purchase_time'] = pd.to_datetime(self.df['purchase_time'])
        self.df['signup_time'] = pd.to_datetime(self.df['signup_time'])
        self.df['time_latency'] = (self.df['purchase_time'] - self.df['signup_time']).dt.total_seconds()/60/60
        #self.df['time_latency'] = (datetime.strptime(self.purchase_time, '%m/%d/%y %H:%M') - datetime.strptime(self.signup_time, '%m/%d/%y %H:%M')).seconds/60/60
        return self.df
    
    def fit_transform(self):
        return self.fit().transform()

### Merge data
class mergeMultipleDataframes():
    def __init__(self, dfs, key, method):
        self.dfs = dfs
        self.key = key
        self.method = method
        
    def fit(self):
        return self
    
    def transform(self):
        return reduce(lambda  left, right: pd.merge(left, right, on = self.key, how=self.method), self.dfs)
    
    def fit_transform(self):
        return self.fit().transform()

### Create features using helper methods above
class createFeatures():
    def __init__(self, path_to_analysis_dataset):
        self.path_to_analysis_dataset = path_to_analysis_dataset
    
    def fit(self):
        return self
    
    def transform(self):
        ## Drop unnamed column
        analysis_df = pd.read_csv(self.path_to_analysis_dataset).drop('Unnamed: 0', axis = 1)
        
        ## Convert signup and purchase times to pandas datetime
        analysis_df.signup_time = pd.to_datetime(analysis_df.signup_time, format = '%m/%d/%Y %H:%M')
        analysis_df.purchase_time = pd.to_datetime(analysis_df.purchase_time, format = '%m/%d/%Y %H:%M')
        
        ## Fill missing values with NA
        analysis_df = analysis_df.fillna('NA')
        
        ## Calucate fraud ratios for the columns below, and save each to a different dataframe
        var_list = ['device_id', 'country', 'age', 'sex', 'source', 'browser']
        d = {}
        for item in var_list:
            d['fraud_by_' + item] = calculateRatioFraud(analysis_df, item).fit_transform()
            
        ## Calculate latency between sign-up and purchase time
        latency_df = calculateTimeLatency(analysis_df).fit_transform()
    
        ## Merge all features
        feature_df = mergeMultipleDataframes([d['fraud_by_device_id'], d['fraud_by_country'], 
                                              d['fraud_by_sex'], d['fraud_by_age'], 
                                              d['fraud_by_browser'], d['fraud_by_source'], 
                                              analysis_df[['user_id', 'purchase_value', 'class']],
                                              latency_df[['user_id', 'time_latency']]
                                              ], key = ['user_id'], method = 'outer').fit_transform()
        
        ## Encode categorical features
        df_cat = MultiColumnLabelEncoder(feature_df[['country', 'sex', 'browser', 'source']])\
        .fit_transform(feature_df[['country', 'sex', 'browser', 'source']])    
    
        return pd.concat([feature_df.drop(['country', 'sex', 'browser', 'source'], axis = 1), df_cat], axis = 1).set_index(['user_id', 'device_id'])
    
    def fit_transform(self):
        return self.fit().transform()
    
    
    

"""
def create_features(path_to_analysis_dataset):
    '''
    Args: 
        path to analysis dataset
        
    Output:
        Dataframe transforms raw data into specific feature elements ready to be used for classfication
    '''
    analysis_df = pd.read_csv(path_to_analysis_dataset)\
    .drop('Unnamed: 0', axis = 1)
    
    ## Convert signup and purchase times to pandas datetime
    analysis_df.signup_time = pd.to_datetime(analysis_df.signup_time, format = '%m/%d/%Y %H:%M')
    analysis_df.purchase_time = pd.to_datetime(analysis_df.purchase_time, format = '%m/%d/%Y %H:%M')
    
    ## Fill missing values with NA
    analysis_df = analysis_df.fillna('NA')
    
    ## Calucate fraud ratios
    var_list = ['device_id', 'country', 'age', 'sex', 'source', 'browser']
    d = {}
    for item in var_list:
        d['fraud_by_' + item] = calculateRatioFraud(analysis_df, item).fit_transform()
    
    ## Calculate latency between sign-up and purchase time
    latency_df = calculateTimeLatency(analysis_df).fit_transform()
    
    ## Merge all features
    feature_df = mergeMultipleDataframes([d['fraud_by_device_id'], d['fraud_by_country'], 
                                          d['fraud_by_sex'], d['fraud_by_age'], 
                                          d['fraud_by_browser'], d['fraud_by_source'], 
                                          analysis_df[['user_id', 'purchase_value', 'class']],
                                          latency_df[['user_id', 'time_latency']]
                                         ], key = ['user_id'], method = 'outer').fit_transform()
    
    df_cat = MultiColumnLabelEncoder(feature_df[['country', 'sex', 'browser', 'source']])\
    .fit_transform(feature_df[['country', 'sex', 'browser', 'source']])
    return pd.concat([feature_df.drop(['country', 'sex', 'browser', 'source'], axis = 1), df_cat], axis = 1).set_index(['user_id', 'device_id'])
"""
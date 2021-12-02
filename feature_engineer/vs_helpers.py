# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 09:17:07 2020

@author: zjc10
"""
import numpy as np 
from utils.decorators import Timer
###########################
#DATE PROCSESSING FUNCTIONS
###########################
def auto_convert_dates(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception: 
                pass
    return df 


def date_to_int(x,shift=None, interval = 'M'):
    if not shift: 
        shift = 0
    d = np.datetime6(x,interval).astype(int)
    return int(d+shift)


def int_to_date(x, interval = 'M'):
    return np.datetime4(np.datetime64(x,interval),interval)


##############################
######VARIABLE TRANSFORMS#####
##############################
@Timer
def var_transforms(df, date_var = None,exclude_vars = None, max_lag = 8):
    """
    var_transforms takes a input dataframe of timeseries data (ex. macro economic drivers)
    and generates time dependent transformations for use in downstream time series modeling

    Args:
        df ([dataframe]): dataframe of time series data with a date variable to indicate chronological order
        date_var ([datetime], required): name of the variable in the dataset representing the date of observations 
        exclude_vars ([list], optional): list of variables to be excluded from lag generation. Defaults to None.
        max_lag (int, optional): max number of lagged variables to genearte. Defaults to 8, with a minimum floor of 4 to ensure yoy transformations can be generated .

    Returns:
        [dataframe]: dataframe of all original input observations and transformed variables 
    """
    
    #date variable must be provided (TRANSFORMS ASSUME QUARTERLY DATE INTERVALS)
    assert (date_var is not None), "A DATE Variable must be provided for sorting and generating lagged transformations"

    #if no variables specified to be excluded from generation of lags , set to empty list 
    if exclude_vars is None: 
        exclude_vars = []

    #variables should be sorted in ascending order of date 
    econ = df.sort_values(by=date_var).copy()
    
    #find numeric columns
    numcols = econ._get_numeric_data().columns

    #create lagged variables of ALL FIELDS in data (require at least 4 lags to calc yoy changes)
    for col in np.setdiff1d(df.columns,exclude_vars):
        for lag in range(1,max(4,max_lag)):
            econ[f'{col}_lag{lag}'] = econ[col].shift(lag)
    
    #create ratios and differences 
    for col in numcols: 

        #yoy 
        econ[f'{col}_yoyrat'] = econ[col] / econ[f'{col}_lag4']

        #yoy %change 
        econ[f'{col}_yoypct'] =  econ[f'{col}_yoyrat']  - 1 

        #yoy diff
        econ[f'{col}_yoydif'] =  econ[col] -  econ[f'{col}_lag4']
        
        #QoQ %change
        econ[f'{col}_qoqrat'] = econ[col] / econ[f'{col}_lag1']

        #lag QoQ %change 
        econ[f'{col}_qoqratlag1'] = econ[f'{col}_lag1']  /  econ[f'{col}_lag2'] 
        
        #QoQ diff 
        econ[f'{col}_qoqdif'] = econ[col] - econ[f'{col}_lag1']
        

        #log transform 
        econ[f'{col}_log'] = np.log(econ[col])

        #sqrt transform 
        econ[f'{col}_sqrt'] = np.sqrt(econ[col])


        #lagged yoy
        econ[f'{col}_yoypctlag1'] = (econ[f'{col}_lag1']  / econ[f'{col}_lag5']) -1  

        #lagged yoy diff    
        econ[f'{col}_yoydiflag1'] = econ[f'{col}_lag1']  - econ[f'{col}_lag5']

        #lagged QoQ pct change
        econ[f'{col}_qoqpctlag1'] = ( econ[f'{col}_lag1'] / econ[f'{col}_lag2'] ) -1 

        #twice lagged QoQ pct change 
        econ[f'{col}_qoqpctlag2'] = ( econ[f'{col}_lag2'] / econ[f'{col}_lag3'] ) -1 

        #log transform YoY (YoY log diff)
        econ[f'{col}_yoyld'] = np.log(econ[f'{col}_yoyrat'])

        #log transform QoQ (QoQ log diff)
        econ[f'{col}_qoqld'] = np.log(econ[f'{col}_qoqrat'])

        #lagged QoQ Log diff 
        econ[f'{col}_qoqldlag1'] = np.log(econ[f'{col}_qoqratlag1'])

    return econ


# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 09:10:55 2020

@author: zjc10
"""


import pandas as pd 
import h2o 
from feature_engineer.vs_helpers import var_transforms
from utils.loggers import init_logging

#create session logger
logger = init_logging(session_only = True)

#initalize h2o session
h2o.init()


#data paths 
dom_econ_path = "C:/Users/zjc10/Desktop/Projects/data/econ/Historic_Domestic.csv"
int_econ_path = "C:/Users/zjc10/Desktop/Projects/data/econ/Historic_International.csv"

#read data 
dom_econ = pd.read_csv(dom_econ_path)
int_econ = pd.read_csv(int_econ_path)


#convert date 
dom_econ['date'] = pd.to_datetime(['-'.join(x.split()[::1]) for x in dom_econ['Date']])
int_econ['date'] = pd.to_datetime(['-'.join(x.split()[::1]) for x in int_econ['Date']])


#transform date to int
dom_econ_trans = var_transforms(dom_econ, date_var ='date' , exclude_vars = None, max_lag = 8)
int_econ_trans = var_transforms(int_econ, date_var ='date' , exclude_vars = None, max_lag = 8)
logger.info(dom_econ_trans.shape)


#shut down h2o 
h2o.shutdown(prompt=False)
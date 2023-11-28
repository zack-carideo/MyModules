#Utility functions for working with datetime objects 
import pandas as pd 
import numpy as np 
from datetime import date


def str2date(year: int,month: int,day: int,interval: str ='M')->np.datetime64:
    """create a numpy datetime64 date from set of input parameters"""
    return np.datetime64(f'{year}-{month}-{day}',interval)

def date2date64(series: pd.Series) -> pd.Series:

    """convert a datetime object to a datetime64 object

    Args:
        series (pd.Series): a series of datetime values 

    Returns:
        pd.Series: a series of datetime64 values 
    """
    
    return pd.to_datetime(series)


def date_to_int(x: date  , shift=None, interval ='M'):
    """convert a datetime value to an integer


    Args:
        dte (datetime): datetime value to convert to int
        shift (int, optional): periods to shift the date by. Defaults to None. + values shift date forward by the specified interval 'M' by default
        interval (str, optional): interval do you want to use when converting the date to integer, i.e 1-> 2 can represent 1 month, 1 quarter, etc.... Defaults to 'M'.
    """
    if not shift:
        shift = 0 
    
    dte_int = np.datetime64(x,interval).astype(int)
    return int(dte_int+shift) 


def int_to_date(int_dte: int, interval: str= 'M'):
    """convert integer date represetnation back to datetime64

    Args:
        int_dte (int): integer representation of datetime object 
        interval (str, optional): interval to use when converting interger back to date. Defaults to 'M'. (should align with method used in date_to_int)
    """
    return np.datetime64(np.datetime64(int_dte, interval),interval)



def array_date_convert(np_array: np.array, date2int: bool = False):
    """ convert numpy array of dates to integers, or numpy array of integer dates back to datetimes
    @param np_array: array of dates to convert
    @param date2int: bool flag to indicate weather or not the dates should be converted into integers or if intergers are being converted back to dates
    @return: np.array() of dates in either integer form (date2int=True) or datetime form (date2int = False)
    """
    if date2int:
        return np.array(np_array,dtype='datetime64[M]').astype(int)
    else:
        return np.array(np_array, dtype='datetime64[M]')

import statsmodels.api as sm
import pandas as pd
import numpy as np
from utils.dates import str2date,date2date64
from typing import Union

"""Functions for deseasonalizing econonomic or other time series 
Notes
    - Only apply deasonalization on the RAW LEVEL SERIES (i.e USGDP, Total_Sales) becuase this is the series that contains seasonality (never apply to a differenced series because you have masked seasonality) 
    - X13 will only operatre correctly when there are no negative values in the time series (aka you are looking at levels and not rates of change) 

Resources 
    - https://stackoverflow.com/questions/43457938/how-to-get-predictions-using-x-13-arima-in-python-statsmodels
    - https://stackoverflow.com/questions/36945324/python-statsmodels-x13-arima-analysis-attributeerror-dict-object-has-no-att
"""



def x13_seasonal_adjust(data: Union[str,pd.DataFrame]
                        , seasonal_col: str = None
                        , date_col: str = None
                        , fcst_periods: int = 0
                        , x13_binary_path: str = None
                        , verbose = False
                        ):
    """function to deseasonalize a time series using the census bearuas x13 procedure
    @param data: path to data to load , or a pandas dataframe
    @param seasonal_col: the columns we want to model , aka the column we want to deseasonalize
    @param date_col: the column in the dataframe containing datetime avlues, this will be used as a datetime index for x13
    @param fcst_periods: number of future periods you want to forecast the seasonalized series for (seasonal_col)
    @param x13_binary_path: path to the x13 binary, if not provided x13 path must be part of the environment variables or on the users path variable
    @return pd.series: deseasonalized time series for the seasonal_col being modeled
    """

    #check inputs
    if isinstance(data,str):
        dta = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        dta = data.copy()
    else:
        raise Exception('must provide path to data or dataframe')

    #verify the column being deseasonalized is not a constant and does not contain negative values (aka you are de-seasonalizing levels)
    if min(dta[seasonal_col])<0:
        raise Exception('YOU CANNOT DE-SEASONALIZE A SERIES WTIH NEGATIVE VALUES, GO BACK AND VERIFY THAT THE SERIES BEING DESEASONALIZED IS LEVEL , NOT THE period over period CHANGE')

    #check date col is date
    if not all(dta[date_col].map(type) == pd.Timestamp):
        raise Exception(f'date column must be of type np.datetime64, you provided: {dta[date_col].map(type).unique()}')
    else:
        #ensure datetime64 is of month interval
        dta[date_col] = dta[date_col].apply(lambda x: np.datetime64(x,'M'))

    #create datetime in
    # Zdex from date col
    dta.set_index(date_col,inplace=True)

    #identify optimal order of seasonal series
    res = sm.tsa.x13_arima_select_order(dta[seasonal_col], x12path = x13_binary_path)
    print(f'regular unit root differencing: {res.order}'
          , f'seasonal order differencing: {res.sorder}'
          )

    #generate
    results = sm.tsa.x13_arima_analysis(
        dta[seasonal_col]  #series to model , must be a pandas series with datetime index or an arraay with start and freq specified
        , x12path = x13_binary_path #path to x13 binary used to execute seasonal adjustment
        , outlier= False #identify and fix outliers
        , trading= True #do trading days impact the seasonality of the input series
        , forecast_periods = fcst_periods #number of periods to forecast forward
    )

    if verbose:
        return results
    else:
        return results.seasadj


def plot_sa_results(results):
    #plot decomposed time series
    fig = results.plot()
    fig.set_size_inches(12, 5)
    fig.tight_layout()


#EXAMPLE RUN
if __name__ == '__main__':
    # inputs
    data = 'C:/Users/zjc10/Desktop/Projects/data/econ/Historic_Domestic.csv'
    x13_binary_path = 'C:/Users/zjc10/Desktop/Utils/winx13_V2.5/WinX13/x13as/x13as'
    seasonal_col = 'House Price Index (Level)'
    date_col = 'Date'
    fcst_periods = 5  # number of periods to forecast the seasonally adjusted series forward

    # preprocessing for test data
    data = pd.read_csv(data)
    q_map = {'q1': '03', 'q2': '06', 'q3': '09', 'q4': '12'}
    data[date_col] = data[date_col].apply(
        lambda x: str2date(str(x.split(' ')[0])
                           , str(q_map[x.split(' ')[1].lower()]), '01', interval='M')
    )

    data[date_col] = data[date_col].apply(lambda x: np.datetime64(x, 'M'))

    data[date_col] = date2date64(data[date_col])

    results = x13_seasonal_adjust(data
                        , seasonal_col=seasonal_col
                        , date_col=date_col
                        , fcst_periods=fcst_periods
                        , x13_binary_path=x13_binary_path
                        , verbose = True
                        )

    #plot the decomposed series
    plot_sa_results(results)
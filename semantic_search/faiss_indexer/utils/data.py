import pandas as pd 
import logging 

#set up basic logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
logger =  logging.getLogger(__name__)


def load_data(df_path)-> pd.DataFrame:
    
    """Normalize column names of input dataframe
    Args:
        df_path (_type_): path to dataframe to use in downstream tasks that will be normalized 
    Returns:
        pd.DataFrame: dataframe with normalized column names 
    """
    
    df = pd.read_csv(df_path)
    df.columns = [x.lower().replace(' ','_') for x in list(df)]
    return df 
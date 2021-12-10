# Utilty functions for reading and writing data 
from utils.decorators import Timer 
import pandas as pd
import logging,yaml,os

logger = logging.getLogger()

@Timer
def open_file(file_path):
    """general function to open a file 

    Args:
        file_path (str): path to file 

    """

    logger.info(os.path.splitext(file_path)[-1])
    if os.path.splitext(file_path)[-1] == '.parquet':
        df = pd.read_parquet(file_path)
    elif os.path.splitext(file_path)[-1] == '.csv':
        logger.info('read csv')
        df = pd.read_csv(file_path)
    elif os.path.splitext(file_path)[-1] == '.xlsx':
        logger.info('read xlsx')
        df = pd.read_xlsx(file_path)
    elif os.path.splitext(file_path)[-1] in ['.pkl','.pickle']:
        logger.info('read pickle')
        df = pd.read_pickle(file_path)
    else:
        raise Exception(f'i dont know how to read this file extension: {os.path.splitext(file_path)[-1]}')

    return df




#load a yaml file into python as a dictionary
def load_yaml(yaml_path:str):
    """[utility to load yaml file]

    Args:
        yaml_path (str): [path to yaml file to parse]

    Returns:
        [dict[anything]]: [dic containing parsed yaml info (values can be any python object)]
    """
    with open(yaml_path,'r')as f:
        yaml_info = yaml.full_load(f)
    return yaml_info

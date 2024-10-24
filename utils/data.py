# Utilty functions for reading and writing data 
from utils.decorators import Timer 
import logging,yaml,os
import pandas as pd
import numpy as np 
from sklearn.model_selection import StratifiedKFold
import requests , zipfile, io


logger = logging.getLogger()

@Timer
def open_file(file_path):
    """general function to open a file with pandas

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

#list all files in a directory
import pathlib
def get_filepaths(directory_root: str):
    """

    @param directory_root: the root to the directory path
    @return: generator of filepaths present in directory_root
    """
    for filepath in pathlib.Path(directory_root).glob('**/*'):
        yield filepath.absolute()
        


#generate k cv datasets for training evaluation 
def stratifiedkfold(y,*,n_splits=5, shuffle=True):
    
    """
    Stratified K-Folds cross-validator 
    Return the indices of the train and validation folds
    
    NOTE: python seed used to ensure reproducibility, it is either set locally or globally via PYTHONHASHSEED envvar  
    """
    SEED = 123 if os.environ.get('PYTHONHASHSEED') is None else int(os.environ['PYTHONHASHSEED'])
    
    skf = StratifiedKFold(n_splits= n_splits, random_state=SEED, shuffle=shuffle)
    splits = list(skf.split(X=np.zeros(len(y)),y=y))
    folds = {}
    
    for idx, (train_idx, val_idx) in enumerate(splits):
        folds[idx] = {'train':train_idx , 'val':val_idx}
    
    return folds, skf



def download_and_unzip(url: str, destination_folder: str):
    
    """
    Downloads a file from the given URL and extracts its contents to the specified destination folder.

    Args:
        url (str): The URL of the file to download.
        destination_folder (str): The path to the folder where the contents of the zip file will be extracted.

    Returns:
        None

    Raises:
        None
    """

    # Send a GET request to download the file
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:

        # Read the content of the response
        content = response.content

        # Create a file-like object from the response content
        file = io.BytesIO(content)

        # Extract the contents of the zip file
        with zipfile.ZipFile(file, 'r') as zip_ref:
            zip_ref.extractall(destination_folder)
    else:
        print("Failed to download the file.")
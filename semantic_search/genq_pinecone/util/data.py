import logging
from datasets import load_dataset
from pathlib import Path 

#set up basic logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
logger =  logging.getLogger(__name__)

def load_data(df_path: str = None
              , load_from_directory=True
              , hf_dataset_name: str = 'squad'
              , split:str ='train'):
    """utility function to load huggyface datasets

    Args:
        hf_dataset_name (str, optional): 
        split (str, optional): 
    """
    if load_from_directory:
        
        assert df_path is not None, 'must provide df directory location and filename if loading data from directory'
        df_path_ = Path(df_path)
        assert df_path_.is_file(), 'you must pass the full path of the dataset you want to load'
               
        return load_dataset(
            str(df_path_.suffix[1:])
            , data_files = str(df_path_)
        )
    
    else:
            
        return load_dataset(hf_dataset_name
                        , split = 'train'
                        )
    

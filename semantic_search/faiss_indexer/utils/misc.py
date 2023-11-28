import os , sys, random, yaml
import numpy as np 
from types import SimpleNamespace
from pathlib import Path
from typing import Tuple 

def check_if_file_exists(file_path: Path) -> Tuple[bool,str]:
    """
    check if file exists
    Args:
        file_path (Path): path to file  

    Returns:
        Tuple[bool,str]: True/False on file existing, message stating file existence at path 
    """
    
    #check if file exists 
    assert isinstance(file_path,Path), 'a path variable must be passed, not a string path aka wrap your input in pathlib.Path(your_file_path)'
    exist = file_path.is_file()
    
    if exist: 
        message = f"File DOES exist:\n\t {file_path}"
    else:
        message = f"file DOES NOT Exist: \n\t {file_path}"
    return exist, message 

def debugger_is_active() -> bool:
    """Return if the debugger is currently active"""
    return hasattr(sys, 'gettrace') and sys.gettrace() is not None

def seed_all(seed: int = 42) -> None:
    """
    seed everything
    Args:
        seed (int, optional): seed numbner, defaults to 42
    """

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    return 

class RecursiveNamespace(SimpleNamespace):
    """
    Extending SimpleNamespace for Nested Dictionaries
    # https://dev.to/taqkarim/extending-simplenamespace-for-nested-dictionaries-58e8

    Args:
        SimpleNamespace (_type_): Base class is SimpleNamespace

    Returns:
        _type_: A simple class for nested dictionaries
    """
    @staticmethod
    def map_entry(entry):
        if isinstance(entry, dict):
            return RecursiveNamespace(**entry)

        return entry

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, val in kwargs.items():
            if type(val) == dict:
                setattr(self, key, RecursiveNamespace(**val))
            elif type(val) == list:
                setattr(self, key, list(map(self.map_entry, val)))


class LoadCFG: 
    """Load a YAML Configuration File 
    This file contains the input and output parameters/variables for the framework 
    """
    def __init__(self
                , filename: str 
                , *
                , base_dir: str='./cfgs/' 
                , return_namespace: bool = True
                ):
        
        #path to file to load 
        self.filename = filename 
        
        #directory of the file to load 
        self.base_dir = Path(base_dir)
        
        #File Path
        self.file_path = self.base_dir / self.filename
        
        #return a namespace object
        self.return_namespace = return_namespace

    def file_exists(self) -> Tuple[bool,str]:
        """check if yaml file exists 

        Returns:
            Tuple[bool,str]: True/False on file existing
        """
        return check_if_file_exists(file_path = self.file_path)

    def load(self, * , check_exists:bool = True):
        """LOAD YAML FILE 

        Args:
            filename(str): Name of file to load 
            check_exists (bool, optional): _description_. Defaults to True.
            
        Raises: 
            ValueError: Error if the YAML file does not exist 
        Returns: 
            _type_: YAML file as either a Namespace Object or Dictionary 
        """
        
        #check if YAML config exists
        if check_exists: 
            exist, message = self.file_exists()
            if exist: 
                print(message)
            else:
                print(f'{message}')
                raise ValueError(message)
        
        #load yaml file 
        with open(self.file_path,'r') as file: 
            cfg = yaml.safe_load(file)
        if self.return_namespace: 
            cfg = RecursiveNamespace(**cfg)
        else: 
            cfg = cfg 
        return cfg 
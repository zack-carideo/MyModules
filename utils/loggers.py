#general use logger setup
import logging 
import sys 
import datetime 
from utils.decorators import Timer

def init_logging(session_only = True, *args, **kwargs) -> logging.Logger : 

    """initalize base logger 

    Args:
        session_only : if True, logging is to be used only within the current interactive session, and no log needs to be saved to file. this will enable the log to print to the interactive shell
    Returns:
        logger
    """

    if session_only is True:
        logging.basicConfig(stream=sys.stdout, level = logging.INFO)   

    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)


    return logger


def add_handler(logger, log_use = 'test', log_out_path=None):
    """add_handler is used to add a file handler to save log info down to. 

    Args:
        logger ([logging.logger()]): an existing logger that accepts file handler
        log_use (str, optional): the function or process the log is being used for. Defaults to 'test'.
        log_out_path ([str], optional): where you want to save the file to. Defaults to None.

    Returns:
        [logger]: returns input logger with expanded file handler, all logging will be saved to the file specified. 
    """
    #create info to uniquly id logfile
    suffix = '_' + datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    logfile = log_out_path + "/" + f"{log_use}_{suffix}.log"

    #create file handler to pass log file to 
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.INFO)

    #create formatter and add it to the handler 
    formatter = logging.Formatter('%(asctime)s - %(levelname)-8s - %(process)d -%(message)s', datefmt = '%d-%b-%y %H:%M:%S')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.info(f'log file name: {logfile}')
    return logger 


@Timer
def example_log_setup(log_use = 'test', session_only = False, log_out_path = 'C:/Users/zjc10/Downloads'):

    """
    wrapper to  initalize and add file handler to logger
    logger will record all activity within the active python session   
    """
    
    #initalize a base logger 
    logger = init_logging(session_only = session_only)
    logger.info('your mom')
    #add file handler if you are not just printing log to the terminal/python log
    if session_only is False:
        assert log_out_path  is not None, logger.info('specify directory to save output log, ex. C:/Users/zjc10/Downloads')
        logger.info(f'log file will be saved to : {log_out_path}')
        logger = add_handler(logger, log_use = log_use, log_out_path = log_out_path)

    return logger


if __name__ == "__main__.py":
    logger = example_log_setup(log_use = 'shell_call', session_only = True)
    logger.info('running example code for illistrating the use of loggers')
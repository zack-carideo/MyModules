import pyspark , os
from typing import List , Union 

import logging 
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)


def spark_init(name: str
    , config: dict = None
    , queue: str = 'default'
    , n_parts: int = None
    , master: str = 'local'
    , hdfs_namespace: str = 'cecldev'
    , enable_arrow: bool = True 
    , *args, **kwargs
    ) -> pyspark.sql.session.SparkSession:
    """_summary_

    Args:
        name (str): descriptive string that will be used as the spark session application name. 
        config(dic,optional): dictionary of spark configuration parameters to manage resources(executor_mem, executor_cores, driver_mem, driver_cores)
        queue (str, optional): the Queue that you want to submit your spark job to. Defaults to 'edl-dev-pyfrm-q-comm'.
        n_parts (int, optional): number of partitions you want to split each job into(if job contains aggregations, map and reduce operations must be used to get correct results across partitions, otherwise you will be aggregating WITHIN each partition , not across ALL partitions). Defaults to None.
        master (str, optional): orchastration manager of spark, if 'yarn' try to use yarn, if local spin up locally managed spark session. Defaults to 'local'.
        hdfs_namespace (str, optional): if running spark with yarn (master= 'yarn') on HDFS this indicates what HDFS environment you are in . Defaults to 'cecldev'.
        enable_arrow (bool, optional): do you want to enable pyspark with arrow, this can reduce i/o spoeeds. Defaults to True.

    Returns:
        pyspark.sql.session.SparkSession: active pyspark session upon which we use to distribute all downstream operations in parrallell
    """

    #verify environemnt variables are set prior to initalization
    assert os.getenv('CLASSPATH',None) is not None, f'you must specify the environment variable CLASSPATH to use arrow'
    assert os.getenv('LD_LIBRARY_PATH',None) is not None, f'you must specify the environment variable LD_LIBRARY_PATH to use arrow'

    #set app name and generate baseline spark config
    appname = f'spark_{name}'
    conf = pyspark.SparkConf().setAppName(appname)


    #check if a config was passed with specified resources 
    if isinstance(config, dict):
        
        check_keys = ['executor_mem','executor_cores','driver_mem','driver_cores'] \
            if n_parts is None else ['executor_mem','executor_cores','driver_mem','driver_cores','executor_instances']
       
        config = {k:v for k,v in config.items() if v is not None}

        assert not any([k not in check_keys for k in config.keys()]) , f'only the following config attributes can be configured{check_keys}'
        logger.info(f'initalizing spark with custom spark cluster configuration: {config}')

    elif config is None:
        config = {k:v for k,v in {
            'executor_mem':'4g'
            , 'executor_cores': 2
            , 'driver_mem':'10g'
            , 'driver_cores': 4
            , 'executor_instances':n_parts
        }.items() if k in check_keys}

    #specify how to manage the spark session(local or using yarn)
    if master.startswith('local'):
        conf.setMaster('local[*]')
    else:
        conf.setMaster('yarn')
        conf.set('spark.yarn.queue', queue)
        conf.set('spark.driver.memory', config['driver_mem'])
        conf.set('spark.driver.cores', config['driver_cores'])
        conf.set('spark.executor.memory', config['executor_mem'])
        conf.set('spark.executor.cores', config['executor_cores'])

        if n_parts is not None:
            conf.set('spark.executor.instances' , config['executor_instances'])
   
    if enable_arrow:
        conf.set('spark.sql.parquet.mergeSchema','false')
        conf.set('spark.hadoop.parquet.enable.summary-metadata','false')

    #set basic conf settings
    conf.set('spark.exeutorEnv.CLASSPATH' , os.environ['CLASSPATH'])
    conf.set('spark.exeutorEnv.LD_LIBRARY_PATH' , os.environ['LD_LIBRARY_PATH'])

    #set environment variables required for spark 
    os.environ['HDFS_NAMESPACE'] = hdfs_namespace
    os.environ['SPARK_YARN_STG_DIR'] = os.environ.get('SPARK_YARN_STG_DIR' , 'hdfs://' + os.environ['HDFS_NAMESPACE'] + '/user/' + os.environ['USER'] )

    conf.set('spark.yarn.stagingDir', os.environ['SPARK_YARN_STG_DIR'])

    #build spark session 
    spark_session = pyspark.sql.SparkSession.builder.config(conf=conf).getOrCreate()
    application_id = spark_session.sparkCOntext.applicationId 
    logger.info(f"Application ID: {application_id}\n SparkWebUI: {spark_session.sparkContext.uiWebUrl}")
    return spark_session


#start a spark cluster and pass any seralized objects needed for run to each worker node 
def start_cluster(name : str 
                , sparkconfig: Union[dict,None] = None
                , queue: str = 'edl-dev-pyfrm-q-comm'
                , n_parts: int = None
                , master: str = 'yarn'
                , hdfs_namespace: str = 'cecldev'
                , zip_file_list:List = None
                , enable_arrow: bool  = False
                ):

    #start spark session 
    spark = spark_init(name
        , config = sparkconfig
        , queue = queue
        , n_parts = n_parts 
        , master = master 
        , hdfs_namespace = hdfs_namespace
        , enable_arrow = enable_arrow
        )


    #generate spark context 
    sc = spark.sparkContext

    #pass any zipped resources required to run code to each spark worker (only need to pass to spark session, and it will handle distribution to workers)
    if zip_file_list is not None:
        for f in zip_file_list:
            sc.addPyFile(f)
    
    return spark, sc 

    


def is_spark_context_active(sc: pyspark.SparkContext):
    """check if a spark context is associated with an active spark serssion 

    Args:
        sc (pyspark.SparkContext): _description_

    Returns:
        _type_: _description_
    """
    return sc._jsc.sc().isStopped()

def stop_spark():
    """wrapper to call to stop spark 
    """
    spark.stop()

####DATAFRAME TO ELASTIC FORMATTING 
import numpy as np
import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch import helpers
from text_processing.text_preprocessing import df_to_json

#filter columns from a dfrow that has been converted into a geneartor 
def filterKeys(document_gen,vars2use):
    return {key:document_gen[key] for key in vars2use}

#generate douments in elastic format 
#yeild is used to allow the function to hand off records to Elastic API only when it asks for them 
def doc_generator(df,index_name, docIdVar, vars2use = None):
    """
    Overview: 
        provide a dict with specified values to elastic bulk api 

    Inputs: 
        df : dataframe to push to elastic 
        index_name: name of index you want to push data to (database)
        docIdVar : id variable in input dataframe 
        vars2use : list of variables from input df we want to push to elastic 
    
    Yield parameters:
        _index: the database name
        _type: the table name (this is now _doc for 7.1 and cannot be any other value) 
        _id: elastic unique ID (not the same as 'id' field from df)
        _source: the document to be saved (you could also simply use document.to_dit())
        raise StopIteration: raise exception when generator is empty
    """

    df_iter = df.iterrows()
    for index, document in df_iter:
        yield {
                "_index": index_name,
                "_type": "_doc",
                "_id" : f"{document[docIdVar]+str(index)}",
                "_source": filterKeys(document,vars2use),
            }
    raise StopIteration

if __name__ == "__main__":
    #STEP 0: spin up elastic clusters (3 of them)
        #cd elasticsearch-7.4.1\bin 
        #.\elasticsearch.bat
        #.\elasticsearch.bat
        #.\elasticsearch.bat

    #STEP 1: Load df to push to index 
    df = pd.read_pickle("C:\\Users\\zjc10\\Desktop\\Projects\\data\\news\\webhose_news\\webhose_df.pickle")

    #STEP 2: MISSING DATA CHECK 
    #ELASTIC CANNOT HANDLE NAN 
    if df.isnull().sum().sum() > 0:
        print("STOP AND FIX MISSING DATA")        

    #STEP 3: Push docs to elastic 
    es_client = Elasticsearch(http_compress=True)
    vars2use = ['key','date','title','author']
    helpers.bulk(es_client,doc_generator(df, "webhose",'type',vars2use=vars2use))

    #STEP 4: show details about index just uploaded
    #curl -X GET "localhost:9200/webhose/_stats?pretty"
    #curl -X GET "localhost:9200/webhose/_doc/ ?pretty"
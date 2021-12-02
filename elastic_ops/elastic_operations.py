#elastic operations 
####DATAFRAME TO ELASTIC FORMATTING 
import numpy as np
import pandas as pd
import json 
import sys
from elasticsearch import Elasticsearch
from elasticsearch import helpers
from text_processing.text_preprocessing import df_to_json

#function to check connection to verify existing elastic connection is up and running 
def check_connection(es_client):
    try:
        info = es_client.indices.get_alias("*")
        print('sucessfully connected to elastic')
        print(info)
    except ConnectionError as c:
        print("Connection Error: {}".format(c))

class elastic_ops():
    def __init__(self,host='localhost',port=9200):

        #create binding to elastic default instance
        self.es_client = Elasticsearch([{'host':host, 'port':port, 'http_compress':True}])

        #verify connection is active before initalizing class 
        check_connection(self.es_client)
    

    ####################
    ######QUERY OPS#####
    ####################
    def search(self, index_name, search):
        res = self.es_client.search(index=index_name, body=search)
        return res

    def match_query(self,index='webhose',query='vaping',field='title',sort='date'):
        return self.es_client.search(
            index, 
            {
                'query':{
                    'match':{
                        field:query,
                    },
                },
                "sort":{
                    sort:'desc'
                }
            }
        )

    def del_index(self,index_name=None):
        self.es_client.indices.delete(index=index_name, ignore = [400,404])

    def load_and_check(self,static_news_pickle = None, vars2use=NOne, docIdVar='id'):

        #load 
        df = pd.read_pickle(static_news_pickle)

        if not vars2use:
            raise Exception("Please specify list of variables to push to elastic(vars2use)")

        if df[vars2use].isnull().sum().sum()>0:
            raise Exception("Elastic cannot handle missing data, please clean , impute , or use placeholders to remove NAs before loading to elastic")

        #subset of data to load into elastic
        df_for_elastic = df[vars2use+[docIdVar]]

        return df_for_elastic

    

    #filter columns from a dfrow that has been converted into a geneartor 
    def filterKeys(self,document_gen,vars2use):
        return {key:document_gen[key] for key in vars2use}

    #generate douments in elastic format 
    #yeild is used to allow the function to hand off records to Elastic API only when it asks for them 
    def doc_generator(self, df=None,index_name=None, docIdVar=None, vars2use = None):
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


    def PushData2Index(self,static_news_pickle=None,index_name = None, docIdVar=None, vars2use=None):
         
         #if vars2used passed via cli, it is inturpreted as a string
         #must strip list operators [] and recreate list from strings
        if type(vars2use).__name__=='str':
            vars2use = [val for val in map(str,vars2use.strip('[]').split(','))]

        if not static_news_pickle or not index_name or not vars2use:
            raise Exception('ALL INPUT PARAMETERS MUST BE SPECIFIED')

        #load and check 
        df2push = self.load_and_check(static_news_pickle=static_news_pickle, vars2use = vars2use, docIdVar = docIdVar)
        print(df2push.shape)

        #push data to elastic 
        helpers.bulk(
            self.es_client,
            self.doc_generator(df = df2push, index_name = index_name, docIdVar = docIdVar, vars2use = vars2use))


if __name__=="__main__":
    if len(sys.argv)!=5:
        print("ERROR: must correctlyu specify input variables to push to elastic")
    else:
        elastic_ops().PushData2Index(static_news_pickle=sys.argv[1],index_name = sys.argv[2],docIdVar = sys.argv[3], vars2use = sys.argv[4])

    #STEP 0: spin up elastic clusters (3 of them)
        #cd elasticsearch-7.4.1\bin 
        #.\elasticsearch.bat
        #.\elasticsearch.bat
        #.\elasticsearch.bat

    #STATIC NEWS FILE LOCATION
    #df = pd.read_pickle("C:\\Users\\zjc10\\Desktop\\Projects\\data\\news\\webhose_news\\webhose_df.pickle")

    #POST RUN: show details about index just uploaded
    #curl -X GET "localhost:9200/webhose/_stats?pretty"
    #curl -X GET "localhost:9200/webhose/_doc/ ?pretty"
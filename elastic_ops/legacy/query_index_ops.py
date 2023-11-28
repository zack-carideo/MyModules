###ELASTIC SEARCH OPS###
#_search API will be leveraged for quering data 
#index to query : webhose
#curl -X GET "localhost:9200/webose/_search?pretty" -H 'Content-Type: application/json' -d'{"query": { "match_all": {} }, "sort": [{ "date": "asc" }]}'
from elasticsearch import Elasticsearch
from elasticsearch import helpers
from text_processing.text_preprocessing import df_to_json
import logging
import json 

#create connection to local es instance
def connect_elasticsearch():
    _es = None
    _es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
    if _es.ping():
        print('Yay Connect')
    else:
        print('Awww it could not connect!')
    return _es

#setup sarch function 
#pass index, and search critera 
def search(es_object, index_name, search):
    res = es_object.search(index=index_name, body=search)


if __name__ == '__main__':
    logging.basicConfig(level=logging.ERROR)
    es = connect_elasticsearch()
    if es is not None:

        search_object = {
                        'query': {
                            'match':{
                                'title':{
                                    'query':'Nintendo'
                                    }
                                }
                            }
                        }
        search(es,'webhose',json.dumps(search_object))


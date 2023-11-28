#https://github.com/AIAnytime/How-to-implement-a-better-RAG/blob/main/lotr.ipynb
# Import the necessary modules
#chromadb (vector database)
import chromadb, os, sys, bs4


# Define the list of retrievers
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.retrievers.merger_retriever import MergerRetriever

from pathlib import Path
from os.path import expanduser

#get data to test demo with 
import bs4
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

#config 
#testing chroma db access
db_dir = "/home/zjc1002/Mounts/data/"
llm_dir = "/home/zjc1002/Mounts/llms/"


db_info = [{'db_name':'mini'
           , 'model_name':"sentence-transformers/all-MiniLM-L6-v2"
           , "model_kwargs": {"device":"cuda"}
           , "collection_meta":{"hnsw:space": "cosine"}
           , "persist_directory": f"{db_dir}/mini_tst"
           , "search_type": "similarity"
           , "search_kwargs": {"k":3, "include_metadata": True}

           }
           , {'db_name':'miniqa'
           , 'model_name':"sentence-transformers/multi-qa-MiniLM-L6-dot-v1"
           , "model_kwargs": {"device":"cuda"}
           , "collection_meta":{"hnsw:space": "cosine"}
           , "persist_directory": f"{db_dir}/miniqa_tst"
           , "search_type": "similarity"
           , "search_kwargs": {"k":3, "include_metadata": True}
           }
           , {'db_name':'bge'
           , 'model_name':"BAAI/bge-large-en-v1.5"
           , "model_kwargs": {"device":"cuda"}
           , "collection_meta":{"hnsw:space": "cosine"}
           , "persist_directory": f"{db_dir}/bge_tst"
           , "search_type": "similarity"
           , "search_kwargs": {"k":3, "include_metadata": True}
           }
           ]

doc_split_info = {'chunk_size':1000
                  , 'chunk_overlap':200}


query_list = ["Is there any framework available to tackle the climate change?","what is huggingGPT?"]

#Define Sample Web Based Loader 
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)

#load and split sample docs 
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=doc_split_info['chunk_size']
                                               , chunk_overlap=doc_split_info["chunk_overlap"])
splits = text_splitter.split_documents(docs)
print(len(splits))



#chromadb 
#load embeddings to use as retrievers
#use lots of small models to reduce bias
embeddings_ = { _db["db_name"]:  HuggingFaceEmbeddings(model_name=_db['model_name']
                                    , model_kwargs = _db['model_kwargs']
                                    , cache_folder=llm_dir)
                                    for _db in db_info}

#doc stores (Chroma)
vector_stores = {_db["db_name"]: Chroma.from_documents(splits
                                     , embeddings_[_db['db_name']]
                                     , collection_metadata=_db['collection_meta']
                                     , persist_directory=_db['persist_directory'])
                                     for _db in db_info}

#load the vector indecies
vec_indcies = {_db["db_name"]: Chroma(persist_directory= _db['persist_directory']
                                      , embedding_function=embeddings_[_db['db_name']])
                                      for _db in db_info
}


#generate retrievers for vector indcies      
vecindex_retrievers = {_db["db_name"]: vec_indcies[_db['db_name']].as_retriever(
    search_type = _db["search_type"]
    , search_kwargs = _db['search_kwargs']
    )
    for _db in db_info
    }


#Generate LORD OF ALL RETRIEVERS (aka an ensemble of retrievers)
lotr = MergerRetriever(retrievers=[retrvr for retrvr in vecindex_retrievers.values()])


#generate responses for original input queries
for _query in query_list: 
    for chunks in lotr.get_relevant_documents(_query):
        print(chunks.page_content)


                                                                                                                 
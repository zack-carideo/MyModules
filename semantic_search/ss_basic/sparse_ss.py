import string,  tqdm 
import pandas as pd
import numpy as np 
from sklearn.feature_extraction import _stop_words
from rank_bm25 import BM25Okapi
from typing import List, Set

def bm25_tokenizer(text: str
                   , stop_words: Set[str] = None
                   , punctuations: str = None
                   ) -> List[str]:
    
    tokenized_doc = []
    for token in text.lower().split():
        token = token.strip(punctuations) if punctuations else token

        if stop_words: 
            if len(token) > 0 and token not in stop_words:
                tokenized_doc.append(token)
        else: 
            if len(token) > 0:
                tokenized_doc.append(token)

    return tokenized_doc


class bm25Search():
    """
    A class to perform BM25 search on a given corpus of documents.
    Attributes:
    -----------
    top_k : int
        The number of top results to return for each query.
    run_id : str
        An identifier for the search run.
    stop_words : Set[str]
        A set of stop words to be removed from the documents and queries.
    punctuations : str
        A string of punctuation characters to be removed from the documents and queries.
    corpus : List[str]
        The corpus of documents to be searched.
    tokenized_corpus : List[List[str]]
        The tokenized version of the corpus.
    bm25 : BM25Okapi
        The BM25 index built from the tokenized corpus.
        
    Methods:
    --------
    __init__: 
        Initializes the bm25Search object with the given parameters and builds the BM25 index.
    search(query: str) -> List[int]: 
        Searches the BM25 index for the given query and returns the top k results.
    search_all(queries: List[str]) -> List[List[int]]:
        Searches the BM25 index for all the given queries and returns the top k results for each query.
        The method combines static query information with dynamic search results and returns a list of dictionaries.
        Each dictionary contains:
            - 'Query_Index': The index of the query.
            - 'Query_Text': The text of the query.
            - 'Run_ID': The run identifier.
            - 'Chunk_ID': A static value of 1.
            - 'Similarity_Score': The similarity scores of the top k results.
            - 'Doc_ID': The document IDs of the top k results.
            - 'Search_Result': The actual text of the top k results.
    """
    
    def __init__(self
                 , corpus: List[str]
                 , stop_words: Set[str] = None
                 , punctuations: str = None
                 , top_k: int = 5
                 , run_id: str = 'zjc_test'
                 ):
        
        self.top_k = top_k
        self.run_id = run_id
        self.stop_words = stop_words
        self.punctuations = punctuations
        self.corpus = corpus
        self.tokenized_corpus = [bm25_tokenizer(doc
                                                ,  punctuations=punctuations
                                                , stop_words=stop_words) 
                                                for doc in tqdm.tqdm(corpus)]
        
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
    #search bm25 index on query and return top k results
    def search(self, query: str) -> List[int]:

        #tokenize query
        tokenized_query = bm25_tokenizer(query
                                         , punctuations=self.punctuations
                                         , stop_words=self.stop_words)
        
        #get bm25 scores
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        #sort scores(descending)
        sorted_doc_ids = np.argsort(doc_scores)[::-1]

        return {  'Similarity_Score':doc_scores[sorted_doc_ids][:self.top_k] 
                , 'Doc_ID':sorted_doc_ids[:self.top_k]
                , 'Search_Result':[self.corpus[idx] for idx in sorted_doc_ids[:self.top_k]]
                }
    
    #search bm25 index on all queries and return top k results
    def search_all(self, queries: List[str]) -> List[List[int]]:

        #combine static query information with dynamic search results
        return [{
            **{'Query_Index':[idx for x in range(0,self.top_k)]
               , 'Query_Text':[query for x in range(0,self.top_k)]
                ,'Run_ID':[self.run_id for x in range(0,self.top_k)]
                , 'Chunk_ID':[1 for x in range(0,self.top_k)]
                },
                 **self.search(query) }
                 for idx,query in enumerate(tqdm.tqdm(queries))]
    

if __name__ == '__main__':



    #Note: only use preprocessing if strings are very long or if you have a lot of data
    punctuations = string.punctuation
    stop_words = _stop_words.ENGLISH_STOP_WORDS


    _queries = ["quick brown fox"
                ,"The quick fox is quick"
                , "lazy dog"
                , "quick fox"
                , "lazy dog"
                , "zack wack"]

    _corpus_text = ["The quick brown fox jumps over the lazy dog.",
        "A quick brown dog outpaces a quick fox.",
        "The quick fox is quick",
        "The dog is lazy.",
        "who zack is wack."
    ]

    #initiate bm25 instance
    _bm25 = bm25Search(
        corpus = _corpus_text
        , stop_words = None
        , punctuations = punctuations
        , top_k = 2
    )

    #search for top k results and return dataframe
    pd.concat([pd.DataFrame(x) for x in _bm25.search_all(_queries)])
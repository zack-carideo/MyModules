import re
import itertools
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.parsing.preprocessing import preprocess_documents
from gensim.parsing.preprocessing import preprocess_string

class doc2vec_docSim():

    def __init__(self
                 , df
                 , text_col_2_index
                 , index_col=None
                 , dm = None
                 , hs = None
                 , dm_mean = None
                 , dm_concat = None
                 , dbow_words = None
                 , vector_size = None
                 , window = None
                 , shrink_window=None
                 , min_count = 1
                 , alpha = .03
                 , min_alpha = .01
                 , epochs = 10
):

        self._df = df
        self._text_col_2_index = text_col_2_index
        self._index_col = index_col
        self._corpus = df[text_col_2_index].values

        self.doc2vec_min_count = min_count
        self.dm = dm
        self.hs = hs
        self.dm_mean = dm_mean
        self.dm_concat = dm_concat
        self.dbow_words = dbow_words
        self.vector_size = vector_size
        self.window = window
        self.shrink_window = shrink_window
        self.alpha = alpha
        self.min_alpha = min_alpha
        self.epochs = epochs    

        self._clean_corpus = None
        self._dictionary = None 
        self._bow_corpus = None
        self._tfidf = None
        self._index_fitted = False
        self._index = None 

        #generate init param settings for doc2vec
        self._params = {k:v for k,v in locals().items() if k 
                        not in ['self', 'df','text_col_2_index', 'index_col','text_col_2_index'] and v is not None}
        
        #generate index after inializing all inputs 
        self.generate_index()

    def preprocess_corpus(self):

        """
        Preprocesses the text corpus by applying tokenization and other preprocessing steps.
        """

        self._clean_corpus = preprocess_documents(self._corpus)
    

    def tag_doc(self):
        
        """
        Converts the preprocessed corpus into tagged word tokenized lists.
        """

        assert self._clean_corpus is not None, "Corpus is not preprocessed"
        assert len(self._clean_corpus)==len(self._df), "Corpus length does not match dataframe length"

        if self._index_col:
            self._tagged_corpus = [TaggedDocument(t[0],[t[1]]) for t in zip(self._clean_corpus
                                                                          ,self._df[self._index_col])]
        else: 
            self._tagged_corpus = [TaggedDocument(d, [i]) for i, d in
                                   enumerate(self._clean_corpus)]


    def generate_index(self):
        """
        Generates the similarity index for the Doc2Vec model.
        """
        self.preprocess_corpus()
        self.tag_doc()


        self._model = Doc2Vec(self._tagged_corpus
                              ,**self._params
                              
                              )
        self._index_fitted = True
    
    def get_similarities(self, query, topn=5):

        """
        Generates the similarity index for the Doc2Vec model.
        """
        
        assert self._index_fitted, "Index has not been fitted"
        assert self._model, "Model has not been fitted"
        
        test_doc_vec = self._model.infer_vector(
            preprocess_string(query)
        )
        
        sims = self._model.dv.most_similar(positive=[test_doc_vec])
        
        return sims[:topn]
    
    def search_index(self, query, topn=5, return_df=False):
        """
        Returns the top n most similar documents to the given query.

        Parameters:
        -----------
        query : str
            The query document.
        topn : int, optional
            The number of similar documents to return. Default is 5.

        Returns:
        --------
        list
            A list of dictionaries containing the similar documents and their similarity scores.
        """
        assert self._index_fitted, 'Index you want to search has not been fitted'
        assert self._model, 'Model has not been fitted'

        #get similarity scores using doc2vec
        matches = self.get_similarities(query, topn=5)

        if self._index_col:
            output = [(self._df.iloc[i[0]][[self._text_col_2_index, self._index_col]].to_dict()
                       , {'sim': i[1],'rank':idx}) for idx,i in enumerate(matches)]
            
        else: 
            output = [(self._df.iloc[i[0]][[self._text_col_2_index]].to_dict()
                       , {'sim': i[1],'rank':idx}) for idx, i in enumerate(matches)]    

        if return_df:
            return pd.DataFrame([{**{k: v for k, v in t[0].items()}
                                  , **{k: v for k, v in t[1].items()}} for t in output])
        else:
            return [{**{k: v for k, v in t[0].items()}
                     , **{k: v for k, v in t[1].items()}} for t in output]


#1. create a test datasets to evaluate search methods with 
#2. 
# Create a dataframe with random sentences

if __name__=='__main__':


    data = {
        'index1': [i for i in range(0,5)],
        'index2': [i+1 for i in range(0,5)],
        'target': [1, 0, 1, 0, 1],
        'lob1': ['a', 'b', 'c', 'd', 'e'],
        'lob2': ['a', 'b', 'h', 'i', 'e'],
        'text1': ['The quick brown fox jumps over the lazy dog', 'Hello world', 'I love programming', 'Python is awesome', 'Data science is interesting'],
        'text2': ['The quick brown dog jumps on the log.', 'Goodbye world', 'I enjoy coding', 'Python is powerful', 'Machine learning is fascinating']
    }
    #create dataframe 
    df = pd.DataFrame(data)


    #use a test query to identify which combination of hyperparms best represents the query
    query= 'The quick brown fox jumps over the lazy dog'

    #hyperparam grid
    dm = [1, 0]
    vector_size = [10, 20, 50, 100, 200]
    window = [1, 2, 3, 4]
    dm_mean = [0,1]
    dm_concat = [0,1]
    dbow_words = [0,1]
    hs = [1, 0]

    paramsList = [{'dm': item[0]
                , 'vector_size': item[1]
                , 'window': item[2]
                , 'hs': item[3]
                , 'dm': item[4]
                , 'dm_mean':item[5]
                , 'dm_concat': item[6]
                , 'dbow_words': item[7]
                } for item in
                    list(itertools.product(*[dm
                                            , vector_size
                                            , window
                                            , hs
                                            , dm 
                                            , dm_mean
                                            , dm_concat
                                            , dbow_words
                                            ]))
                ]


    #loop over hyperparms and query index for each set
    res = []
    for _parmas in paramsList:
        d2v_indexer = doc2vec_docSim(df, 'text1',index_col = 'index1',**_parmas
        )

        res.append(d2v_indexer.search_index(query, topn=2, return_df=True)
                .assign(**_parmas)
                )
        
    #Rsults dataframe from all experiments 
    res_df = pd.concat(res)










import re, json, gensim, itertools,pickle
import pandas as pd
from pathlib import Path
from gensim.models import TfidfModel
from gensim.similarities import MatrixSimilarity
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.parsing.preprocessing import preprocess_documents
from gensim.parsing.preprocessing import preprocess_string
from wonderwords import RandomSentence

s = RandomSentence()

class tfidf_docSim():
    """
    A class for performing TF-IDF based document similarity search.

    Parameters:
    -----------
    df : pd.DataFrame
        The input dataframe containing the text data.
    text_col_2_index : str
        The name of the column in the dataframe that contains the text data.
    index_col : str, optional
        The name of the column in the dataframe that contains the index values. Default is None.
    tfidf_param : str, optional
        The parameter for TF-IDF calculation. Default is 'Lpb'.

    Methods:
    --------
    preprocess_corpus():
        Preprocesses the text corpus by applying tokenization and other preprocessing steps.
    doc2bow():
        Converts the preprocessed corpus into bag-of-words representation.
    gen_tfidf():
        Generates the TF-IDF model using the bag-of-words corpus.
    generate_index():
        Generates the similarity index for the TF-IDF model.
    get_similarities(query, topn=5):
        Returns the top n most similar documents to the given query.
    search_index(query, topn=5, return_df=False):
        Searches the index for documents similar to the given query.

    Example:
    --------
    # Create a dataframe with random sentences
    data_size = 4
    data = {
        'index1': [i for i in range(0,data_size*5)],
        'index2': [i+1 for i in range(0,data_size*5)],
        'target': data_size*[1, 0, 1, 0, 1],
        'lob1': data_size*['a', 'b', 'c', 'd', 'e'],
        'lob2': data_size*['a', 'b', 'h', 'i', 'e'],
        'text1': data_size*[f"{s.sentence()}.{s.sentence()}",s.sentence(),s.sentence(), s.sentence(),s.sentence()],
        'text2': data_size*[f"{s.sentence()}.{s.sentence()}",s.sentence(),s.sentence(), s.sentence(),s.sentence()]
    }

    # Create a tfidf_docSim object
    tfidf_indexer = tfidf_docSim(df=data, text_col_2_index='text1', tfidf_param='Lpb')

    # Search for similar documents
    results = tfidf_indexer.search_index('The precious tracksuit tears pill', topn=5, return_df=True)
    print(results)
    """

    def __init__(self
                 , index_col=None
                 , tfidf_param='Lpb'
                 , model_output_dir=None):
        """
        Initializes the tfidf_docSim object.

        Parameters:
        -----------
        df : pd.DataFrame
            The input dataframe containing the text data.

        index_col : str, optional
            The name of the column in the dataframe that contains the index values. Default is None.
        tfidf_param : str, optional
            The parameter for TF-IDF calculation. Default is 'Lpb'.
        """
        self._model_output_dir = model_output_dir
        self._index_col = index_col
        self._tfidf_param = tfidf_param

        self._df = None 
        self._text_col_2_index = None        
        self._clean_corpus = None
        self._dictionary = None 
        self._bow_corpus = None
        self._tfidf = None
        self._index_fitted = False
        self._index = None 

        #self._corpus = df[text_col_2_index].values
        #self.generate_index()

    def create_corpus(self, df, text_col_2_index):
        """
        Creates a corpus from the given dataframe and text column.

        Parameters:
        -----------
        df : pd.DataFrame
            The input dataframe containing the text data.
        text_col_2_index : str
            The name of the column in the dataframe that contains the text data.
        """
        self._df = df
        self._text_col_2_index = text_col_2_index
        self._corpus = df[text_col_2_index].values
        

    def preprocess_corpus(self):
        """
        Preprocesses the text corpus by applying tokenization and other preprocessing steps.
        """
        self._clean_corpus = preprocess_documents(self._corpus)
    
    def doc2bow(self):
        """
        Converts the preprocessed corpus into bag-of-words representation.
        """
        assert self._clean_corpus is not None, "Corpus is not preprocessed"
        self._dictionary = gensim.corpora.Dictionary(self._clean_corpus)
        self._bow_corpus = [self._dictionary.doc2bow(doc) for doc in self._clean_corpus]
    
    def gen_tfidf(self):
        """
        Generates the TF-IDF model using the bag-of-words corpus.
        """
        assert self._bow_corpus is not None, "Corpus has not been converted to bow"
        self._tfidf = TfidfModel(corpus=self._bow_corpus 
                                 , dictionary=self._dictionary
                                 , smartirs=self._tfidf_param)
 
    def generate_index(self, df, text_col_2_index):
        """
        Generates the similarity index for the TF-IDF model.
        """
        self.create_corpus( df, text_col_2_index)
        self.preprocess_corpus()
        self.doc2bow()
        self.gen_tfidf()
        self._index = MatrixSimilarity(self._tfidf[self._bow_corpus])
        self._index_fitted = True
        return self 
        
    
    def get_similarities(self, query, topn=5):
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
        assert self._index, 'Index you want to search has not been fitted'
        
        query = preprocess_string(query)
        query_bow = self._dictionary.doc2bow(query)
        query_tfidf = self._tfidf[query_bow]
        sims = self._index[query_tfidf]
        topsims = sorted(enumerate(sims), key=lambda item: -item[1])[:topn]
        return topsims
    
    def search_index(self, query, topn=5, return_df=False):
        """
        Searches the index for documents similar to the given query.

        Parameters:
        -----------
        query : str
            The query document.
        topn : int, optional
            The number of similar documents to return. Default is 5.
        return_df : bool, optional
            Whether to return the results as a pandas DataFrame. Default is False.

        Returns:
        --------
        list or pd.DataFrame
            A list of dictionaries or a pandas DataFrame containing the similar documents and their similarity scores.
        """
        matches = self.get_similarities(query, topn)

        if self._index_col:
            output = [(self._df.iloc[i[0]][[self._text_col_2_index, self._index_col]].to_dict(), {'sim': i[1]}) for i in matches]
        else: 
            output = [(self._df.iloc[i[0]][[self._text_col_2_index]].to_dict(), {'sim': i[1]}) for i in matches]    
        
        if return_df:
            return pd.DataFrame([{**{k: v for k, v in t[0].items()}, **{k: v for k, v in t[1].items()}} for t in output])
        else:
            return [{**{k: v for k, v in t[0].items()}, **{k: v for k, v in t[1].items()}} for t in output]


    def save_model(self, model_output_dir):
        """
        Saves the TF-IDF model to the given output directory.

        Parameters:
        -----------
        model_output_dir : str
            The output directory to save the model.
        """
        assert self._tfidf, 'Model has not been fitted'
        assert self._dictionary, 'Model has not been fitted'
        assert self._index, 'Model has not been fitted'

        _out_dir = Path(model_output_dir).as_posix()
        _tfidf_outpath = f"{_out_dir}/tfidf.pkl"
        _dictionary_outpath = f"{_out_dir}/dictionary.model"
        _index_outpath = f"{_out_dir}/index.model"
        _indexed_data_outpath = f"{_out_dir}/indexed_data.pq"
        _class_params_outpath = f"{_out_dir}/class_params.json"

        self._tfidf.save(_tfidf_outpath)
        self._dictionary.save(_dictionary_outpath)
        self._index.save( _index_outpath) 
        self._df.to_parquet(_indexed_data_outpath)

        with open(_class_params_outpath, 'w') as f:
            json.dump({'index_col':self._index_col
                       , 'tfidf_param':self._tfidf_param
                       , 'text_col_2_index':self._text_col_2_index
                       },f)


        return {'tfidf': _tfidf_outpath, 'dictionary': _dictionary_outpath
                , 'index': _index_outpath, 'indexed_data': _indexed_data_outpath
                , 'out_params': _class_params_outpath
                }
    
    def load_model(self, model_output_dir):
        """
        Loads the TF-IDF and gensim index model from the given output directory.

        Parameters:
        -----------
        model_output_dir : str
            The output directory to load the model from.
        """
        _out_dir = Path(model_output_dir).as_posix()
        _tfidf_outpath = f"{_out_dir}/tfidf.pkl"
        _dictionary_outpath = f"{_out_dir}/dictionary.model"
        _index_outpath = f"{_out_dir}/index.model"
        _indexed_data_outpath = f"{_out_dir}/indexed_data.pq"
        _class_params_outpath = f"{_out_dir}/class_params.json"
        
        #load model objects
        self._model_output_dir = model_output_dir
        self._df = pd.read_parquet(_indexed_data_outpath)
        self._tfidf = TfidfModel.load(_tfidf_outpath)
        self._dictionary = gensim.corpora.Dictionary.load(_dictionary_outpath)
        self._index = MatrixSimilarity.load(_index_outpath)
        
        #load class params
        with open(_class_params_outpath, 'r') as f:
            parms = json.load(f)
        self._index_col = parms['index_col']
        self._tfidf_param = parms['tfidf_param']    
        self._text_col_2_index = parms['text_col_2_index']
        
        #create corpus and generate bow corpus
        self._corpus = self._df[self._text_col_2_index].values
        self.preprocess_corpus()
        self._bow_corpus = [self._dictionary.doc2bow(doc) for doc in self._clean_corpus]
        return self 


if __name__ == '__main__':

    model_output_dir = "C:\\Users\\zjc10\\OneDrive\\Desktop\\output\\"
    # Create a dataframe with random sentences
    data_size = 4
    data = {
        'index1': [i for i in range(0,data_size*5)],#data_size*[1, 2, 3, 4, 5],
        'index2': [i+1 for i in range(0,data_size*5)],
        'target': data_size*[1, 0, 1, 0, 1],
        'lob1': data_size*['a', 'b', 'c', 'd', 'e'],
        'lob2': data_size*['a', 'b', 'h', 'i', 'e'],
        'text1': data_size*[f"{s.sentence()}.{s.sentence()}",s.sentence(),s.sentence(), s.sentence(),s.sentence()],#['The quick brown fox jumps over the lazy dog', 'Hello world', 'I love programming', 'Python is awesome', 'Data science is interesting'],
        'text2': data_size*[f"{s.sentence()}.{s.sentence()}",s.sentence(),s.sentence(), s.sentence(),s.sentence()]#['The quick brown dog jumps on the log.', 'Goodbye world', 'I enjoy coding', 'Python is powerful', 'Machine learning is fascinating']
    }

    #create dataframe 
    df = pd.DataFrame(data)

    hp_search = False #do you want to evaluate HP search?
    save_mod = False #do you want to test saving or loading the model (if save_mod is False then a model is loaded from an existing model_output_dir)

    if not hp_search: 

        if save_mod: 
            #quick test 
            #generate index 
            _indexer = tfidf_docSim(index_col='index1', tfidf_param = 'Lpb'
                                    , model_output_dir = model_output_dir
                                    )

            _indexer = _indexer.generate_index(df, 'text1')

            #search the index
            res = _indexer.search_index('The measure frantic drags worried times rejoices.'
                                    , topn=5,return_df=True)
            
            #save model
            _indexer.save_model(model_output_dir)
        else: 
            #load model 

            _indexer = tfidf_docSim()
            _indexer = _indexer.load_model(Path(model_output_dir).as_posix())

            #search the index
            res = _indexer.search_index('The measure frantic drags worried times rejoices.'
                                    , topn=5,return_df=True)

            print(res.head(10))

    else: 
        #set up param grid for hyperparm tuning 
        termfreq = ['b','l']#['b', 'n', 'a', 'l', 'd', 'L']
        docfreq = ['n','f']#['n', 'f', 'p']
        docnorm = ['n','b']#['n', 'c', 'u', 'b']

        param_grid = [''.join(comb) for comb in
                list(itertools.product(*[termfreq,
                                            docfreq,
                                            docnorm]))
                    ]

        #holds results of hp eval
        res = []

        for _params in param_grid: 
            
            try: 
                #generate index 
                _idexer = tfidf_docSim(index_col='index1', tfidf_param = _params
                                    , model_output_dir = model_output_dir
                                    )
                
                _idexer = _idexer.generate_index(df, 'text1')


                #tfidf_indexer = tfidf_docSim(df, 'text1', tfidf_param = _params)

                #search the index 
                res.append(_idexer.search_index('The measure frantic drags worried times rejoices.'
                                        , topn=5,return_df=True).assign(param=_params))

            except:
                print(f"param setup: {_params} failed") 
        res_df = pd.concat(res)
        print(res_df.head(10))
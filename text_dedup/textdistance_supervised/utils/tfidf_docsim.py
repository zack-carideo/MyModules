
import re
import gensim
import itertools
import pandas as pd
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

    def __init__(self, df, text_col_2_index, index_col=None, tfidf_param='Lpb'):
        """
        Initializes the tfidf_docSim object.

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
        """
        self._df = df
        self._text_col_2_index = text_col_2_index
        self._index_col = index_col
        self._corpus = df[text_col_2_index].values
        self._tfidf_param = tfidf_param
        
        self._clean_corpus = None
        self._dictionary = None 
        self._bow_corpus = None
        self._tfidf = None

        self._index_fitted = False
        self._index = None 

        self.generate_index()

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
        self._tfidf = TfidfModel(corpus=self._bow_corpus, dictionary=self._dictionary, smartirs=self._tfidf_param)
 
    def generate_index(self):
        """
        Generates the similarity index for the TF-IDF model.
        """
        self.preprocess_corpus()
        self.doc2bow()
        self.gen_tfidf()
        self._index = MatrixSimilarity(self._tfidf[self._bow_corpus])
        self._index_fitted = True
        
    
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

#generate index 
tfidf_indexer = tfidf_docSim(
                   df
                 , 'text1'
                 , tfidf_param = 'Lpb'
                 )

#search the index 
tfidf_indexer.search_index('The easy bikini rains meatball.	'
                           , topn=5,return_df=True)



import re , itertools, logging , string
import polars as pl 
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.stem.snowball import SnowballStemmer
#logger
logger = logging.getLogger(__name__)
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


def generate_combos(df: pl.DataFrame
                    , index_col1: str
                    , index_col2: str
                    , text_col1: str 
                    , text_col2: str 
                    , target_col: str 
                    )-> pl.DataFrame:
    
    """
    Generates pairwise combinations of text from two columns in a DataFrame.

    Args:
        df (pl.DataFrame): The input DataFrame.
        index_col1 (str): The name of the first index column.
        index_col2 (str): The name of the second index column.
        text_col1 (str): The name of the first text column.
        text_col2 (str): The name of the second text column.

    Returns:
        pl.DataFrame: A DataFrame containing all pairwise combinations of text from the specified columns.

    """

    #list of all pairwise (non symmetric) combinations of text1 and text2 
    idx_combos = pl.DataFrame(list(itertools.product(df[index_col1], df[index_col2]))
                              ).transpose().rename({'column_0':index_col1, 'column_1':index_col2})

    #create a dataframe joining pairwise indexes to incorp actual text values of text1 and text2
    last_join_cols = [v for v in [index_col1, index_col2,target_col] if v is not None]

    df_pairs = idx_combos.join(
                    df.select([index_col1, text_col1]), on=index_col1, how='left'
                    ).join(
                        df.select([index_col2, text_col2]), on=index_col2, how='left'
                        ).join(df.select(last_join_cols), on=[index_col1,index_col2]
                               , how='left')
    
    if target_col is not None: 
        df_pairs = df_pairs.with_columns(pl.col(target_col).fill_null(pl.lit(0)))

    return df_pairs

#text processing for modeling (beyond basic input text formatting)
def clean_sentence(sentence: str
                   , punctuation: set = None
                   , stemmer: SnowballStemmer = None
                   , lower: bool = False
                   , stopwords: list = None
                   ) -> str:
    """
    Cleans a sentence by removing punctuation, converting to lowercase
    , removing stopwords, and applying stemming if specified.

    Args:
        sentence (str): The input sentence to be cleaned.
        punctuation (str, optional): A string containing the punctuation characters to be removed. Defaults to string.punctuation+"\\\\".
        stemmer (object, optional): An object implementing the stemmer interface for word stemming. Defaults to None.
        lower (bool, optional): Flag indicating whether to convert the sentence to lowercase. Defaults to False.
        stopwords (list, optional): A list of stopwords to be removed from the sentence. Defaults to None.

    Returns:
        str: The cleaned sentence.

    """
    sentence = sentence.encode('ascii',errors = 'ignore').decode()

    if punctuation is not None: 
        #sentence=re.sub(f"""[{punctuation}]""",' ',sentence)
        sentence = sentence.translate(str.maketrans("","", string.punctuation))
        sentence = re.sub(' {2,}',' ', sentence)


    #if lower 
    sentence= sentence.lower().strip() if lower else sentence.strip()
    
    #if stopwords 
    if stopwords: 
        sentence = ' '.join([word for word in sentence.split() if word not in stopwords])
    
    #if stem 
    if stemmer: 
        sentence = ' '.join([stemmer.stem(word) for word in sentence.split()])
    return sentence

def clean_text_cols(df: pl.DataFrame
                    , text_cols:list = None
                    , punc = None
                    , stemmer = None
                    , lower_text = False
                    , stopwords = None
                    ) -> pl.DataFrame:
    """
    wrapper for clean_sentence function to clean text columns in a polars DataFrame.
    """
     #clean text 

    for col in text_cols:
        df = df.with_columns(
            pl.col(col).map_elements(lambda x: clean_sentence(x
                                                            , punctuation = punc 
                                                            , stemmer = stemmer 
                                                            , lower = lower_text
                                                            , stopwords = stopwords 
                                                            ) 
                                                            , return_dtype= pl.Utf8()
                                                            ).alias(col))
    
    return df 


#IF YOU WANT TO USE PANDAS AND SKLEARN FOR FULL E2E HERE IS THE CUSTOM TRRANSFORMER FOR TEXT CLEANING
class CleanSentenceTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, punctuation=None, stemmer=None, lower=False, stopwords=None):
        self.punctuation = punctuation
        self.stemmer = stemmer
        self.lower = lower
        self.stopwords = stopwords

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        cleaned_sentences = []
        for sentence in X:
            cleaned_sentence = clean_sentence(sentence, punctuation=self.punctuation, stemmer=self.stemmer, lower=self.lower, stopwords=self.stopwords)
            cleaned_sentences.append(cleaned_sentence)
        return cleaned_sentences






import sys, yaml, os, string, re
import pandas as pd 
import polars as pl
import textdistance as td
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import seaborn as sns
import yake 
import matplotlib.pyplot as plt
import textblob



def get_yake_keywords(x:str, n=5, dedupLim=0.9, dedupFunc='seqm', windowsSize=5, top=20):

    """
    Extracts top N keywords from the input text using the YAKE algorithm.

    Args:
        x (str): The input text.

    Returns:
        list: A list of top N keywords extracted from the input text.

    
    Example:
        text = "The quick brown fox jumps over the lazy dog"
        keywords = custom_kw_extractor.extract_keywords(text)
        print(keywords)
    """

    #yake setup
    custom_kw_extractor = yake.KeywordExtractor(lan='en'
                                                , n=n
                                                , dedupLim=dedupLim
                                                , dedupFunc=dedupFunc
                                                , windowsSize=windowsSize
                                                , top=top 
                                                )

    return ','.join([x[0] for x in custom_kw_extractor.extract_keywords(x)])



#text preprocessing stuff 
PUNC = set(string.punctuation)
STOPWORDS = stopwords.words('english')
STEMMER = SnowballStemmer('english')


#location of data to evalute
df_url = 'https://raw.githubusercontent.com/brmson/dataset-sts/master/data/sts/sick2014/SICK_train.txt'
target_col = 'entailment_judgment'
text_col1 = 'sentence_A'
text_col2 = 'sentence_B'

remove_stopwords = True
remove_punc = True
lower_text = True
stem_text = True 

algo_descriptions = {
    "td.Prefix": "Calculates the longest common prefix length between two strings.",
    "td.Postfix": "Calculates the longest common suffix length between two strings.",
    "td.Length": "Calculates the absolute difference in length between two strings.",
    "td.Cosine": "Measures the cosine similarity between two strings, which is the cosine of the angle between their vector representations.",
    "td.Jaccard": "Measures the Jaccard similarity between two strings, which is the size of the intersection divided by the size of the union of the two sets.",
    "td.Bag": "Compares two strings based on multiset theory, which allows for multiple occurrences of elements.",
    "td.Sorensen": "Measures the Sorensen-Dice coefficient between two strings, which is twice the size of the intersection divided by the sum of the sizes of the two sets.",
    "td.MongeElkan": "Applies a secondary string distance function to all pairs of substrings and averages the results.",
    "td.Overlap": "Measures the overlap coefficient between two strings, which is the size of the intersection divided by the size of the smaller set.",
    "td.Tanimoto": "Measures the Tanimoto coefficient between two strings, which is the size of the intersection divided by the size of the union of the two sets.",
    "td.Tversky": "Measures the Tversky index between two strings, which is a generalization of the Jaccard coefficient.",
    "td.Levenshtein": "Measures the Levenshtein distance between two strings, which is the minimum number of single-character edits (insertions, deletions, or substitutions) required to change one string into the other.",
    "td.Hamming": "Measures the Hamming distance between two strings, which is the number of positions at which the corresponding symbols are different.",
    "td.NeedlemanWunsch": "Implements the Needleman-Wunsch algorithm, which is used in bioinformatics to align protein or nucleotide sequences.",
    "td.SmithWaterman": "Implements the Smith-Waterman algorithm, which is used in bioinformatics to perform local sequence alignment.",
    "td.Gotoh": "Implements the Gotoh algorithm, which is used in bioinformatics to perform sequence alignment with affine gap scoring.",
    "td.StrCmp95": "Implements the Jaro-Winkler string similarity measure, which is designed for comparing short strings like person names.",
    "td.MLIPNS": "Implements the Metric Longest Increasing Subsequence Pseudo-Normalized Similarity, which is a string similarity measure based on the longest increasing subsequence problem.",
    "td.ArithNCD": "Implements the Arithmetic Normalized Compression Distance, which is a measure of the compressibility of a string.",
    "td.BWTRLENCD": "Implements the Burrows-Wheeler Transform Relative Lempel-Ziv Normalized Compression Distance, which is a measure of the compressibility of a string.",
    "td.RLENCD": "Implements the Run-Length Encoding Normalized Compression Distance, which is a measure of the compressibility of a string.",
    "td.ZLIBNCD": "Implements the Zlib Normalized Compression Distance, which is a measure of the compressibility of a string.",
    "td.SqrtNCD": "Implements the Square Root Normalized Compression Distance, which is a measure of the compressibility of a string.",
    "td.EntropyNCD": "Implements the Entropy Normalized Compression Distance, which is a measure of the compressibility of a string.",
    "td.MRA": "Implements the Match Rating Approach, which is a phonetic algorithm developed to assist in the matching of data in areas such as information retrieval and record linkage.",
    "td.Editex": "Implements the Editex algorithm, which is a string distance measure that takes into account phonetic similarities.",
    "td.LCSSeq": "Implements the Longest Common Subsequence Similarity, which is a measure of the longest subsequence common to two strings.",
    "td.LCSStr": "Implements the Longest Common Substring Similarity, which is a measure of the longest substring common to two strings.",
    "td.RatcliffObershelp": "Implements the Ratcliff/Obershelp pattern recognition algorithm, which measures the similarity between two strings.",
    "td.Jaro": "Implements the Jaro string similarity measure, which is designed for comparing short strings like person names.",
    "td.JaroWinkler": "Implements the Jaro-Winkler string similarity measure, which is a variant of the Jaro measure that gives more favorable ratings to strings that match from the beginning."
}


algos = [td.Prefix()
 , td.Postfix()
 , td.Length()
 , td.cosine
 , td.jaccard
 , td.Bag()
 , td.Sorensen()
 , td.MongeElkan()
 , td.Overlap()
 , td.Tanimoto()
 , td.Tversky()
 , td.levenshtein
 , td.hamming
 , td.NeedlemanWunsch()
 , td.SmithWaterman()
 , td.Gotoh()
 , td.StrCmp95()
 , td.MLIPNS()
 , td.ArithNCD()
 , td.BWTRLENCD()
 , td.RLENCD()
 , td.ZLIBNCD()
 , td.SqrtNCD()
 , td.EntropyNCD()
 , td.MRA()
 , td.Editex()
 , td.LCSSeq()
 , td.LCSStr()
 , td.RatcliffObershelp()
 , td.Jaro()
 , td.JaroWinkler()
 ]

def get_data(df_url):
    """
    Retrieves data from the specified URL and returns it as a DataFrame.

    Args:
        df_url (str): The URL of the data file.

    Returns:
        DataFrame: The loaded data.

    """
    #get data 
    df = pl.read_csv(df_url,  separator='\t')
    return df

#2nd level string cleaning 
#text processing for modeling (beyond basic input text formatting)
def clean_sentence(sentence: str
                   , punctuation: str = None
                   , stemmer = None
                   , lower: bool = False
                   , stopwords = None
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
        sentence=re.sub(f'[{punctuation}]',' ',sentence)
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


#load data 
df = get_data(df_url).head(100)

#clean text 
for col in [text_col1, text_col2]:

    df = df.with_columns(
        pl.col(col).map_elements(lambda x: clean_sentence(x
                                                        , punctuation = PUNC if remove_punc else None
                                                        , stemmer = STEMMER if stem_text else None
                                                        , lower = True if lower_text else False
                                                        , stopwords = STOPWORDS if remove_stopwords else None
                                                        ) 
                                                        , return_dtype= pl.Utf8()
                                                        ).alias(f"{col}_cleaned")
        )

    #extract keywords
    df = df.with_columns(
        pl.col(f"{col}_cleaned").map_elements(lambda x: get_yake_keywords(x)
                                              , return_dtype= pl.Utf8()
                                                        ).alias(f"{col}_keywords")
        )
    




#create unifired object with details on  each method
algo_dic_list = []
for idx,algo in enumerate(algos):
    algo_dic_list.append({'colname':f"td.{algo}".split('(')[0]
                          ,'func':algo
                          , 'description':algo_descriptions[f"td.{algo}".split('(')[0]]
                          })

    #compute text similarity
    df = df.with_columns(
        pl.struct([f"{text_col1}_cleaned",f"{text_col2}_cleaned"])\
        .map_elements(lambda x: algo.normalized_similarity(x[f"{text_col1}_cleaned"], x[f"{text_col2}_cleaned"])
                        , return_dtype= pl.Float64()
                        ).alias(f"{algo}".split('(')[0]
        ))
    
    
    #compute text similarity of keywords?
    df = df.with_columns(
        pl.struct([f"{text_col1}_keywords",f"{text_col2}_keywords"])\
        .map_elements(lambda x: algo.normalized_similarity(x[f"{text_col1}_keywords"]
                                                           , x[f"{text_col2}_keywords"])
                        , return_dtype= pl.Float64()
                        ).alias(f"{algo}".split('(')[0]+"_kywrds"
        ))
    
    #compute sentiment of text1 vs text2 (variance in sentiment from textblob)
    df = df.with_columns(
        pl.struct([f"{text_col1}",f"{text_col2}"])\
        .map_elements(lambda x: textblob.TextBlob(x[f"{text_col1}"]
                                                  ).sentiment.polarity - textblob.TextBlob(
                                                      x[f"{text_col2}"]).sentiment.polarity
                        , return_dtype= pl.Float64()
                        ).alias(f"{algo}".split('(')[0]+"_sentiment"
        ))
    
    #compute similarity of ner tags?
    #use sentiment analysis?
    

#go to pandas 
pdf = df.to_pandas()

# Group by index and calculate mean for all numeric columns
numeric_cols = [v for v in pdf.select_dtypes(include='number').columns.tolist() 
                if v not in ['pair_ID','relatedness_score']]

#plot values across all similarity metrics across levels of 'target'
pdf[numeric_cols+[target_col]].groupby(target_col
                                       , dropna=False
                                       ).mean().T.sort_values(by='ENTAILMENT').plot()



# Create box and whisker plot for each variable
for i, col in enumerate(numeric_cols):
    
    sns.displot(x='entailment_judgment'
                , y=col, data=pdf)

plt.tight_layout()
plt.show()



#add logistic regression model to predict target of duplication. 
from sklearn.linear_model import LogisticRegression
#START HERE
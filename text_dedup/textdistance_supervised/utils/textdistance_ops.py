import sys, os, logging , textblob, yake, re
import pandas as pd 
import polars as pl
import textdistance as td
from strsimpy import SIFT4
from rouge import Rouge

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
from utils.text_preprocess import generate_combos, clean_sentence

#logger
logger = logging.getLogger(__name__)
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)



def get_rouge_scores(text1, text2,_rouge=None):
    """
    Computes the ROUGE scores for the given pair of texts.

    Args:
        _rouge (object): The ROUGE object.
        text1 (str): The first text.
        text2 (str): The second text.

    Returns:
        dict: A dictionary containing the ROUGE scores.

    """
    dd = _rouge.get_scores(text1, text2) 
    return {k:v for k,v in dd[0].items()}
    #return ';'.join([f"{k}=={v['f']}" for k,v in dd[0].items()])   
    #return _rouge.get_scores(text1, text2) 



def get_yake_keywords(x:str, n=5, dedupLim=0.9
                      , dedupFunc='seqm', windowsSize=5, top=20):

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



def get_sift4_dist(string1, string2):
    """
    Calculate the SIFT4 similarity between two strings.

    Parameters:
    string1 (str): The first string.
    string2 (str): The second string.

    Returns:
    float: The SIFT4 similarity score between the two strings.
    """

    sift4 = SIFT4(  )
    return sift4.distance(string1, string2)


def generate_features(df: pl.DataFrame
                      , text_col1: str 
                      , text_col2: str 
                      , eval_keywords: bool  = False
                      , td_algos: list = None
                      )-> pl.DataFrame :
    """
    Generate features for text deduplication
    The generate_features() function takes in a DataFrame (df) along with column names (index_col1, index_col2, text_col1, text_col2) and various preprocessing parameters. It computes and returns a new DataFrame with additional features based on the input text columns.
    Here is a detailed breakdown of what the function does:
        4) If eval_keywords is True, it extracts keywords from the cleaned text using the get_yake_keywords() function.
        5) If eval_keywords is True, it calculates the difference in sentiment polarity between the keywords in text_col1 and text_col2 using the sentiment_dif_kw feature.
        6) If eval_keywords is True, it calculates the SIFT4 distance between the keywords in text_col1 and text_col2 using the sift4_dist_kw feature.
        7) It calculates the difference in sentiment polarity between text_col1 and text_col2 using the sentiment_dif feature.
        8) It calculates the SIFT4 distance between text_col1 and text_col2 using the sift4_dist feature.
        9) It calculates the normalized similarity scores between text_col1 and text_col2 using various text distance algorithms from the textdistance library.
        10) If eval_keywords is True, it calculates the normalized similarity scores between the keywords in text_col1 and text_col2 using the same text distance algorithms.
        11) The function returns the updated DataFrame with the added features.      

    Args:
        df (pl.DataFrame): The input DataFrame.
        text_col1 (str): The name of the first text column.
        text_col2 (str): The name of the second text column.
        eval_keywords (bool, optional): Whether to evaluate keywords. Defaults to False.
        td_algos: list, optional): A list of text distance algorithms to use. Defaults to None.
    Returns:
        pl.DataFrame: The updated DataFrame with additional features based on the input text columns.
    """
    
    # generate keywords for each text col if requested 
    for col in [text_col1, text_col2]:

        if eval_keywords:

            #extract keywords
            df = df.with_columns(
                pl.col(f"{col}").map_elements(lambda x: get_yake_keywords(x)
                                                    , return_dtype= pl.Utf8()
                                                                ).alias(f"{col}_keywords")
            )


    #calculate sentiment diff and sift dist of keywords if requested 
    if eval_keywords:

        #sentiment diff on keywords 
        df = df.with_columns(
            pl.struct([f"{text_col1}_keywords",f"{text_col2}_keywords"])\
            .map_elements(lambda x: textblob.TextBlob(x[f"{text_col1}_keywords"]
                                                        ).sentiment.polarity - textblob.TextBlob(
                                                            x[f"{text_col2}_keywords"]).sentiment.polarity
                            , return_dtype= pl.Float64()
                            ).alias(f"sentiment_dif_kw"
            ))
        
        #calc sift dist of keywords
        df = df.with_columns(
            pl.struct([f"{text_col1}_keywords",f"{text_col2}_keywords"])\
            .map_elements(lambda x: get_sift4_dist(x[f"{text_col1}_keywords"],x[f"{text_col2}_keywords"])
                            , return_dtype= pl.Float64()
                            ).alias(f"sift4_dist_kw"
            ))   
            

    #compute text distances
    #compute sentiment of text1 vs text2 (variance in sentiment from textblob)
    df = df.with_columns(
        pl.struct([f"{text_col1}",f"{text_col2}"])\
        .map_elements(lambda x: textblob.TextBlob(x[f"{text_col1}"]
                                                    ).sentiment.polarity - textblob.TextBlob(
                                                        x[f"{text_col2}"]).sentiment.polarity
                        , return_dtype= pl.Float64()
                        ).alias(f"sentiment_dif"
        ))


    #compute sift4_dist socres of text1 vs text2 ()
    df = df.with_columns(
        pl.struct([f"{text_col1}",f"{text_col2}"])\
        .map_elements(lambda x: get_sift4_dist(x[f"{text_col1}"],x[f"{text_col2}"])
                        , return_dtype= pl.Float64()
                        ).alias(f"sift4_dist"
        ))

    
    #compute text distance between text1 and text2
    for idx,algo in enumerate(td_algos):

        if eval_keywords: 
            #compute text similarity of keywords?
            df = df.with_columns(
                pl.struct([f"{text_col1}_keywords",f"{text_col2}_keywords"])\
                .map_elements(lambda x: algo.normalized_similarity(x[f"{text_col1}_keywords"]
                                                                , x[f"{text_col2}_keywords"])
                                , return_dtype= pl.Float64()
                                ).alias(f"{algo}".split('(')[0]+"_kw"
                ))
            
        #compute text similarity
        df = df.with_columns(
            pl.struct([f"{text_col1}",f"{text_col2}"])\
            .map_elements(lambda x: algo.normalized_similarity(x[f"{text_col1}"]
                                                            , x[f"{text_col2}"])
                            , return_dtype= pl.Float64()
                            ).alias(f"{algo}".split('(')[0]
            ))
        
    return df 


#convert to pandas and generate rouges(way easier using pandas)

def pl2pd_and_rouge(df:pl.DataFrame
                    , rouge_metrics:list
                    , _rouge:Rouge
                    , text_col1:str
                    , text_col2:str
                    , eval_keywords:bool = False
                    )-> pd.DataFrame:

    """
    Convert a PySpark DataFrame to a Pandas DataFrame and calculate Rouge scores.

    Args:
        df (pl.DataFrame): The PySpark DataFrame to convert.
        rouge_metrics (list): A list of Rouge metrics to calculate.
        _rouge (Rouge): The Rouge object used for calculating Rouge scores.
        text_col1 (str): The name of the column containing the first text.
        text_col2 (str): The name of the column containing the second text.
        eval_keywords (bool, optional): Whether to calculate Rouge scores for keywords. Defaults to False.

    Returns:
        pd.DataFrame: The Pandas DataFrame with Rouge scores added as columns.
    """

    #go to pandas 
    pdf = df.to_pandas()

    #rouge scores for text
    pdf = pdf.assign(**dict(zip([x.replace('-','_') for x in rouge_metrics]
                        ,zip(*pdf.apply(lambda x: [x['f'] for x in get_rouge_scores(x[f"{text_col1}"]
                                                                                    , x[f"{text_col2}"]
                                                                                    ,_rouge=_rouge).values()], axis=1)))))

    if eval_keywords:
        #rouge scores for keywords
        pdf = pdf.assign(**dict(zip([f"{x.replace('-','_')}_kw" for x in rouge_metrics]
                            ,zip(*pdf.apply(lambda x: [x['f'] for x in get_rouge_scores(x[f"{text_col1}_keywords"]
                                                                                        , x[f"{text_col2}_keywords"]
                                                                                        ,_rouge=_rouge).values()], axis=1)))))

    return pdf 

#main function to generate model df
def gen_model_df(df
                      , text_col1 
                      , text_col2 
                      , eval_keywords = None
                      , td_algos=None
                      , rouge_metrics=None
                      , _rouge=None
                      )-> pd.DataFrame:
    """
    This is a wrapper function to combine generate features and pl2pd_and_rouge functions.
    """
    #generate text distance metrics with polars 
    x = generate_features(df
                        , text_col1 
                        , text_col2 
                        , eval_keywords = eval_keywords
                        , td_algos=td_algos
                        )

    #convert to pandas and calculate rouge score 
    pdf = pl2pd_and_rouge( x.clone()
                        , rouge_metrics
                        , _rouge
                        , text_col1
                        , text_col2
                        , eval_keywords = eval_keywords
                        )

    return pdf
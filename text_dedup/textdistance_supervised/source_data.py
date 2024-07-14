import os, logging, sys, string 
from pathlib import Path
import polars as pl
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from rouge import Rouge
import textdistance as td
from wonderwords import RandomSentence
s = RandomSentence()

#logger
logger = logging.getLogger(__name__)
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)



def create_data(*args,**kwargs)->pl.DataFrame:
    """
    GENERATE DATA LOADING FUNCTION
    for ex. Create a dataframe with random sentences
    """
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

    #create polars dataframe 
    df = pl.DataFrame(data)

    return df 


def preprocess_data(df_inpath: str  = None
                    , index_col1: str = None
                    , index_col2: str = None
                    , text_col1: str = None
                    , text_col2: str = None 
                    , target_col: str = None
                    , punc: set = None
                    , stemmer: SnowballStemmer = None
                    , lower_text: bool = False
                    , stopwords: list = None
                    , eval_keywords: bool = False
                    , td_algos: list = None
                    , rouge_metrics: list = None
                    , _rouge: Rouge = None
                    , final_data_outpath: str = None
                    
                     ):

    #load data
    df = create_data(df_inpath)

    #clean sentences before generating combinations(so we dont have to clean sentences >1 time)
    df = clean_text_cols(df
                            , text_cols=[text_col1, text_col2]
                            , punc=punc
                            , stemmer=stemmer
                            , lower_text=lower_text
                            , stopwords=stopwords
                            )


    #generate dataframme of all combinations of both text columns(by index) 
    #NOTE: if target_col is None, we are assuming preprocessing for scoring and not training
    df = generate_combos(df
                        , index_col1
                        , index_col2
                        , text_col1
                        , text_col2
                        , target_col
                        )


    #generate text distance, sift with polars , convert to pandas and rouge metrics.  
    pdf = gen_model_df(
        df
        , text_col1
        , text_col2
        , eval_keywords = eval_keywords
        , td_algos = td_algos
        , rouge_metrics = rouge_metrics
        , _rouge = _rouge)

    
    if final_data_outpath is None:
        return pdf
    else:
        #save it 
        pdf.to_parquet(final_data_outpath)
        logger.info(f"Data saved to {final_data_outpath}")



if __name__ == '__main__': 

    #CUSTOM MODULES
    project_root = os.path.dirname(os.path.abspath(__file__))
    cfg_path = Path(project_root) / 'configs' / 'config.yaml'

    sys.path.append(project_root)
    from utils.misc import LoadCFG
    from utils.textdistance_ops import  gen_model_df 
    from utils.text_preprocess import generate_combos,clean_text_cols


    #load config file
    CFG = LoadCFG(filename=cfg_path, return_namespace=False).load()
    df_inpath = CFG['data']['df_inpath']
    text_col1 = CFG['data']['text_col1']
    text_col2 = CFG['data']['text_col2']
    index_col1 = CFG['data']['index_col1']
    index_col2 = CFG['data']['index_col2']
    target_col = CFG['data']['target_col']

    #text preprocessing stuff 
    remove_stopwords = CFG['preprocessing']['remove_stopwords']
    remove_punc = CFG['preprocessing']['remove_punc']
    lower_text = CFG['preprocessing']['lower_text']
    stem_text = CFG['preprocessing']['stem_text']    
    eval_keywords = CFG['preprocessing']['eval_keywords']
    train_flag = CFG['model']['train_model']
    td_algos = CFG['model']['distance_metrics']['td_algos']
    rouge_metrics = CFG['model']['distance_metrics']['rouge_metrics']

    #final preproceseed data outpath 
    final_data_outpath = CFG['preprocessing']['df_outpath']

    #rouge object 
    _rouge = Rouge(metrics=rouge_metrics)

    #text distance algorithms
    td_algos = [eval(x) for x in td_algos]

    #preprocessing inputs 
    PUNC = set(string.punctuation)
    STOPWORDS = list(set(stopwords.words('english'))) if remove_stopwords else None
    STEMMER = SnowballStemmer('english') if stem_text else None


    preprocess_data( df_inpath= df_inpath
                    , index_col1=index_col1
                    , index_col2=index_col2
                    , text_col1=text_col1
                    , text_col2 =  text_col2
                    , target_col=  target_col
                    , punc=PUNC
                    , stemmer=STEMMER
                    , lower_text=lower_text
                    , stopwords=STOPWORDS
                    , eval_keywords=eval_keywords
                    , td_algos=td_algos
                    , rouge_metrics=rouge_metrics
                    , _rouge=_rouge
                    , final_data_outpath= final_data_outpath
                     )
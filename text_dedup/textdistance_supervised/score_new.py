
import os, logging, sys, string , joblib
from pathlib import Path
import polars as pl
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from rouge import Rouge
import textdistance as td


if __name__ == '__main__': 

    #
    #CUSTOM MODULES
    project_root = os.path.dirname(os.path.abspath(__file__))
    cfg_path = Path(project_root) / 'configs' / 'config.yaml'
    sys.path.append(project_root)
    from utils.misc import LoadCFG
    from source_data import preprocess_data


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
    model_outdir = CFG['model']['train_params']['model_outdir']
    best_model_path = CFG['model']['train_params']['best_estimator']

    #best model from model selection script 
    best_model = joblib.load(best_model_path)
    #rouge object 
    _rouge = Rouge(metrics=rouge_metrics)

    #text distance algorithms
    td_algos = [eval(x) for x in td_algos]

    #preprocessing inputs 
    PUNC = set(string.punctuation)
    STOPWORDS = list(set(stopwords.words('english'))) if remove_stopwords else None
    STEMMER = SnowballStemmer('english') if stem_text else None

    #columns to exlcude from modelings 
    exclude_cols = [text_col1
                    , text_col2
                    , index_col1
                    , index_col2
                    ,f"{text_col1}_keywords"
                    , f"{text_col2}_keywords"
                    ,target_col]

    #import best model 
    #LEAVE taret_col=None if you are scoring data that does not have a target column
    score_df = preprocess_data( df_inpath= df_inpath
                    , index_col1=index_col1
                    , index_col2=index_col2
                    , text_col1=text_col1
                    , text_col2 =  text_col2
                    , target_col=  None
                    , punc=PUNC
                    , stemmer=STEMMER
                    , lower_text=lower_text
                    , stopwords=STOPWORDS
                    , eval_keywords=eval_keywords
                    , td_algos=td_algos
                    , rouge_metrics=rouge_metrics
                    , _rouge=_rouge
                    , final_data_outpath= None
                        )


    #define x and y
    Xs = [x for x in score_df.columns if x not in exclude_cols]

    X = score_df[Xs]

    #score new 
    import pandas as pd 
    scores = pd.DataFrame(best_model.predict_proba(X))
    scores['predicted'] = best_model.predict(X)




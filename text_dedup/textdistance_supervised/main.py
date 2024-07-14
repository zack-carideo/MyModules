import os
import sys
import string
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
import textdistance as td 
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from rouge import Rouge
from sklearn.model_selection import train_test_split
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
from utils.misc import LoadCFG
from utils.textdistance_ops import  gen_model_df 
from utils.text_preprocess import generate_combos,clean_text_cols
from utils.model_ops import TextSelector, CategoricSelector, NumberSelector, univariant_sk_vs

from wonderwords import RandomSentence
s = RandomSentence()

cfg_path = Path(project_root) / 'configs' / 'config.yaml'
CFG = LoadCFG(filename=cfg_path, return_namespace=False).load()
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

#rouge object 
_rouge = Rouge(metrics=rouge_metrics)

#preprocessing inputs 
PUNC = set(string.punctuation)
STOPWORDS = list(set(stopwords.words('english'))) if remove_stopwords else None
STEMMER = SnowballStemmer('english') if stem_text else None

#text distance algorithms
td_algos = [eval(x) for x in td_algos]

#columns to exlcude from modelings 
exclude_cols = [text_col1, text_col2,f"{text_col1}_cleaned", f"{text_col2}_cleaned",index_col1, index_col2
                ,f"{text_col1}_keywords", f"{text_col2}_keywords",target_col]


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

#clean sentences before generating combinations(so we dont have to clean sentences >1 time)
df = clean_text_cols(df
                          , text_cols=[text_col1,text_col2]
                          , punc=PUNC
                          , stemmer=STEMMER
                          , lower_text=lower_text
                          , stopwords=STOPWORDS
                          , out_col_suffix=None
                          )


#generate dataframme of of all combinations of both text columns(by index) 
df = generate_combos(df, index_col1, index_col2, text_col1, text_col2, target_col)


#generate text distance, sift with polars , convert to pandas and rouge metrics.  
pdf = gen_model_df(
      df
    , text_col1
    , text_col2
    , eval_keywords = eval_keywords
    , td_algos = td_algos
    , rouge_metrics = rouge_metrics
    , _rouge = _rouge)

#define x and y
Xs = [x for x in pdf.columns if x not in exclude_cols]
y = target_col

#split into test and train 
X_train, X_test, y_train, y_test = train_test_split(pdf[Xs+[index_col1,index_col2]]
                                                    , pdf[y]
                                                    , test_size=0.2
                                                    , random_state=42
                                                    , stratify=pdf[y]
                                                    )



#sklearn start 

max_iters = 10 
# Define a pipeline to search for the best combination of PCA truncation
# Define a Standard Scaler to normalize inputs and classifier regularization.
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import  RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier, StackingClassifier, RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import RFECV, VarianceThreshold, mutual_info_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import fbeta_score , make_scorer
import gc
import sklearn.feature_selection as fs 
#basics
scaler = StandardScaler()
pca = PCA()
final_estimator = LogisticRegression()
#final_estimator = XGBClassifier(random_state=13)

#define custom scorer
custom_scorer = make_scorer(
    fbeta_score, beta=1 , average='binary' ,  zero_division=np.nan,) 


#defining individual pipelines to use in stacking classifier 
log_cv = LogisticRegressionCV(cv=3
                              , scoring=custom_scorer
                              )

_rfecv = RFECV(
    estimator = GradientBoostingClassifier(n_estimators=100,max_depth=2, random_state=0)
    , step=1
    , cv = 3
    , scoring = custom_scorer
    , min_features_to_select = 5)


#defining individual pipelines to use in stacking classifier 
logistic = LogisticRegression(max_iter=max_iters, tol=0.05)

feature_selector =   RFECV(
    estimator = GradientBoostingClassifier(n_estimators=100,max_depth=2, random_state=0)
    , step=1
    , cv = StratifiedKFold(5)
    , scoring = custom_scorer
    , min_features_to_select = 3)

logistic_vs = Pipeline(
    steps = [
      ("scaler", scaler)
    , ('vs',feature_selector)
    , ('lr',LogisticRegression(max_iter=max_iters, tol=0.05))])

     
      # set the tolerance to a large value to make the example faster
rf = RandomForestClassifier(n_estimators=100)
gbm = GradientBoostingClassifier(n_estimators=100,learning_rate=1.0,max_depth=1, random_state=0)
mlp = MLPClassifier(max_iter=500,learning_rate = 'adaptive',early_stopping=True)
pca_lr = Pipeline(steps=[("scaler", scaler)
                       , ("pca", pca)
                       , ("logistic", logistic)])


estimators = [
    ('rf', rf),
    ('logistic_vs', logistic_vs),
    ('gbm',gbm),
    ('mlp', mlp),
    ('pca_lr', pca_lr)
]

clf = StackingClassifier(estimators=estimators
                         , final_estimator=final_estimator, cv=StratifiedKFold(5))


model_dict = {

    'logistic_ada': [
        #('variance_check', fs.VarianceThreshold(threshold=(.995)*(1-.995)))
         ('scaler', MinMaxScaler(feature_range=(0,1)))
        , ('univariant_vs', univariant_sk_vs(k=20,min_occurances=1,max_out=20))
        , ('model_vs', fs.SelectFromModel(AdaBoostClassifier(),max_features = 10 , prefit=False))
        , ('clf', LogisticRegressionCV(cv=3, scoring=custom_scorer))
        ]

    , 'logistic_rfecv': [
        #('variance_check', fs.VarianceThreshold(threshold=(.995)*(1-.995)))
         ('scaler', MinMaxScaler(feature_range=(0,1)))
        , ('univariant_vs', univariant_sk_vs(k=20,min_occurances=1,max_out=20))
        , ('model_vs', _rfecv)
        , ('clf', LogisticRegressionCV(cv=3, scoring=custom_scorer))
        ]

    , 'logistic_pca': [
        #('variance_check', fs.VarianceThreshold(threshold=(.995)*(1-.995)))
         ('scaler', MinMaxScaler(feature_range=(0,1)))
        , ('univariant_vs', univariant_sk_vs(k=10,min_occurances=2,max_out=20))
        , ("pca", pca)
        , ('clf', LogisticRegressionCV(cv=3, scoring=custom_scorer))
        ]

    , 'stacked_ensemble': [
        ('clf',clf)       
        ]
    }


#GENERATE PIPELINES FOR EACH MODEL AND FIT TO DATA 
pipe_dic = {key:Pipeline([process for process in value]) for key, value in model_dict.items()}
pipe_fit = {key:pipe.fit(X_train, y_train) for key, pipe in pipe_dic.items()}

for k,v in pipe_fit.items():    
    print(f"{k}:{pipe_fit[k].score(X_train,y_train)}")

#'univariant_vs_k': [5,20]
#, 'univariant_vs__min_occurances': [1,2]
hp_dict = {
    'logistic_ada': {
                      'model_vs__max_features': [10]
                     , 'clf__Cs': [1,2,4]
                     ,'clf__penalty': ['l2',None]
                     }

    , 'logistic_rfecv': {
                          'clf__Cs': [1,2,4]
                         }

    ,'logistic_pca': {
                       'pca__n_components': [5, 15,'mle']
                      , 'pca__whiten': [True , False]
    }
    ,'stacked_ensemble': {'clf_rf__n_estimators': [100, 200, 300]
                          ,'clf_gbm__n_estimators': [100, 200, 300]
                          ,'clf_mlp__hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)]
                          ,'clf_mlp__activation': ['tanh', 'relu']
                          ,'clf_pca_lr__pca__n_components': [5, 15, 30, 'mle']
                              }
}


hp_fit ={}

for k,v in hp_dict.items():
    search = RandomizedSearchCV(pipe_fit[k], v, n_jobs=2)
    search.fit(X_train, y_train)
    hp_fit[k] = search
    print(f"{k}: Best parameter (CV score={search.best_score_}):")
    print(f"{k}")
    print(search.best_params_)
    del search 
    gc.collect()


hps = {'rf__n_estimators': [100, 200, 300]
       , 'gbm__n_estimators': [100, 200, 300]
       , 'mlp__hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)]
       , 'mlp__activation': ['tanh', 'relu']
       , 'pca_lr__pca__n_components': [5, 15, 30, 45, 60,'mle']
       , 'pca_lr__pca__whiten': [True, False]
       , 'pca_lr__logistic__C': np.logspace(-4,0, 1,4)
}


search = RandomizedSearchCV(clf, hps, n_jobs=2,)
search.fit(X_train, y_train)
print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)







#
#stacking classifier 
#
from sklearn.feature_selection import RFECV 
from sklearn.model_selection import StratifiedKFold


estimators = [
    ('rf', rf),
    ('logistic_vs', logistic_vs),
    ('gbm',gbm),
    ('mlp', mlp),
    ('pca_lr', pca_lr)
]

clf = StackingClassifier(estimators=estimators
                         , final_estimator=final_estimator, cv=StratifiedKFold(5))

hps = {'rf__n_estimators': [100, 200, 300]
       , 'gbm__n_estimators': [100, 200, 300]
       , 'mlp__hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)]
       , 'mlp__activation': ['tanh', 'relu']
       , 'pca_lr__pca__n_components': [5, 15, 30, 45, 60,'mle']
       , 'pca_lr__pca__whiten': [True, False]
       , 'pca_lr__logistic__C': np.logspace(-4,0, 1,4)
}


search = RandomizedSearchCV(clf, hps, n_jobs=2,)
search.fit(X_train, y_train)
print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)
clf.fit(X_train, y_train).score(X_test, y_test)




from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()

_input = pd.concat(
    [pdf[[index_col1, text_col1]].drop_duplicates(subset=[index_col1,text_col1]).rename(
        columns={index_col1:'index'
                 , text_col1:'text'})
        , pdf[[index_col2, text_col2]].drop_duplicates(subset=[index_col2,text_col2]).rename(
            columns={index_col2:'index'
                     , text_col2:'text'})
                     ], ignore_index=True)

dtm = vectorizer.fit_transform(_input['text'])
pairwise_similarity = dtm * dtm.transpose()
print(vectorizer.get_feature_names_out())
print(dtm.toarray())
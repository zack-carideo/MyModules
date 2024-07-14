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
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import  RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier, StackingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import fbeta_score , make_scorer
import gc
import sklearn.feature_selection as fs 
from wonderwords import RandomSentence
s = RandomSentence()

#CUSTOM MODULES
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
from utils.misc import LoadCFG
from utils.textdistance_ops import  gen_model_df 
from utils.text_preprocess import generate_combos,clean_text_cols
from utils.model_ops import univariant_sk_vs

#max iterations of logistic training before stopping
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
rouge_metrics = CFG['model']['distance_metrics']['rouge_metrics']

#training details 
n_cv = CFG['model']['train_params']['n_cv']
max_iters = CFG['model']['train_params']['max_iters']
min_features = CFG['model']['train_params']['min_features']
n_estimators = CFG['model']['train_params']['n_estimators']
max_depth = CFG['model']['train_params']['max_depth']
stop_tol = CFG['model']['train_params']['stop_tol']
learning_rate = CFG['model']['train_params']['learning_rate']

#define custom scorer TO USE IN GRID SEARCH and model evaluation
custom_scorer = make_scorer(fbeta_score, beta=1 , average='binary' ,  zero_division=np.nan,) 

#final estimator to use in stacked classifer (it should / can be a more complex model than the base estimators) - I AM USING LOGISTIC FOR PURPOSE OF SPEED FOR ILLISTRATION
final_estimator = eval(CFG['model']['train_params']['final_estimator'])

#columns to exlcude from modelings 
exclude_cols = [text_col1
                , text_col2
                , index_col1
                , index_col2
                ,f"{text_col1}_keywords"
                , f"{text_col2}_keywords"
                ,target_col]
 

#load data
pdf = pd.read_parquet(CFG['preprocessing']['df_outpath'])

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


#recursive feature elimination with cross validation
_rfecv = RFECV(
    estimator = GradientBoostingClassifier(n_estimators=n_estimators
                                           ,max_depth=max_depth
                                           , random_state=0)
    , step=1
    , cv = n_cv
    , scoring = custom_scorer
    , min_features_to_select = min_features)


#defining individual pipelines to use in stacking classifier 
logistic_vs = Pipeline(
    steps = [
            ("scaler", StandardScaler())
            , ('vs',_rfecv)
            , ('lr',LogisticRegression(max_iter=max_iters, tol=stop_tol))])

     
# set the tolerance to a large value to make the example faster
rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=0)
gbm = GradientBoostingClassifier(n_estimators=n_estimators,learning_rate=learning_rate
                                 ,max_depth=max_depth, random_state=0)
mlp = MLPClassifier(max_iter=max_iters,learning_rate = 'adaptive',early_stopping=True)

#speicify the estimators you want to use in the stacking classifier
estimators = [
    ('rf', rf),
    ('logistic_vs', logistic_vs),
    ('gbm',gbm),
    ('mlp', mlp),
]

#create the stacking classifier
clf = StackingClassifier(estimators=estimators
                         , final_estimator=final_estimator
                         , cv=StratifiedKFold(n_cv))


#create pipeline steps for each model you want to test
model_dict = {

    'gbm_ada': [
        #('variance_check', fs.VarianceThreshold(threshold=(.995)*(1-.995)))
        ('scaler', MinMaxScaler(feature_range=(0,1)))
        , ('univariant_vs', univariant_sk_vs(k=30,min_occurances=1,max_out=30))
        , ('model_vs', fs.SelectFromModel(AdaBoostClassifier(),max_features = 30 , prefit=False))
        , ('clf', GradientBoostingClassifier( ))
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
        , ("pca", PCA())
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
    print(f"{k}:{pipe_fit[k].score(X_test,y_test)}")

#
# Hyper Parameter Setup
#
hp_dict = {
    # 'logistic_ada': {
    #                   'model_vs__max_features': [10,20]
    #                   ,'clf__ccp_alpha':[0,.1,.2]
    #                  , 'clf__max_depth': [1,2,3]
    #                  , 'clf__max_features':[2,'auto','sqrt','log2']
    #                  #,'clf__penalty': ['l2',None,'l1']
    #                  }

    # , 'logistic_rfecv': {
    #                       'clf__Cs': [1,2,4]
    #                      }

    # ,'logistic_pca': {
    #                    'pca__n_components': [5, 15,'mle']
    #                   , 'pca__whiten': [True , False]
    # }
    'stacked_ensemble': {'clf__rf__n_estimators': [100]
                          ,'clf__gbm__n_estimators': [100]
                          ,'clf__mlp__hidden_layer_sizes': [(50,25)]
                          ,'clf__mlp__activation': ['tanh', 'relu']
                              }
}



search = RandomizedSearchCV(pipe_fit['stacked_ensemble'], hp_dict['stacked_ensemble'], n_jobs=4)
search.fit(X_train, y_train)
k=1
print(f"{k}: Best parameter (CV score={search.best_score_}):")
print(f"{k}")
print(search.best_params_)

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



search = RandomizedSearchCV(clf, hps, n_jobs=2,)
search.fit(X_train, y_train)
print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)







#
#stacking classifier 
#
from sklearn.feature_selection import RFECV 
from sklearn.model_selection import StratifiedKFold
pca_lr = Pipeline(steps=[("scaler",  StandardScaler()), ("pca", PCA()), ("logistic"
                                                                         , LogisticRegression(max_iter=max_iters
                                                                                              , tol=stop_tol))])



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
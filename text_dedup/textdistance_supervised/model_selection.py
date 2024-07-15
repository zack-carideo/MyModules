

import os, sys, joblib, gc, yaml,logging
import sys
import joblib
import gc
import yaml
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier, StackingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
import sklearn.feature_selection as fs
from sklearn.metrics import fbeta_score, make_scorer

#logger
logger = logging.getLogger(__name__)
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


if __name__ =='__main__':
    #PROJET ROTO 
    project_root = os.path.dirname(os.path.abspath(__file__))

    #CUSTOM MODULES
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

    #training details 
    train_flag = CFG['model']['train_model']
    model_outdir = CFG['model']['train_params']['model_outdir']
    n_cv = CFG['model']['train_params']['n_cv']
    max_iters = CFG['model']['train_params']['max_iters']
    min_features = CFG['model']['train_params']['min_features']
    n_estimators = CFG['model']['train_params']['n_estimators']
    max_depth = CFG['model']['train_params']['max_depth']
    stop_tol = CFG['model']['train_params']['stop_tol']
    learning_rate = CFG['model']['train_params']['learning_rate']

    #columns to exlcude from modelings 
    exclude_cols = [text_col1, text_col2, index_col1, index_col2,f"{text_col1}_keywords"
                    , f"{text_col2}_keywords",target_col]
    

    #define custom scorer TO USE IN GRID SEARCH and model evaluation
    custom_scorer = make_scorer(fbeta_score, beta=1 , average='binary' ,  zero_division=np.nan,) 

    #final estimator to use in stacked classifer (it should / can be a more complex model than the base estimators) - I AM USING LOGISTIC FOR PURPOSE OF SPEED FOR ILLISTRATION
    final_estimator = eval(CFG['model']['train_params']['final_estimator'])


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
    _rfecv = fs.RFECV(
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
    gbm = GradientBoostingClassifier(n_estimators=n_estimators,learning_rate=learning_rate,max_depth=max_depth, random_state=0)
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
            ('scaler', MinMaxScaler(feature_range=(0,1))) 
            , ('variance_check', fs.VarianceThreshold(threshold=(.9995)*(1-.9995))) 
            #, ('univariant_vs', univariant_sk_vs(k=30,min_occurances=1,max_out=30))
            #, ('model_vs', fs.SelectFromModel(AdaBoostClassifier(),max_features = 30 , prefit=False))
            , ('clf', GradientBoostingClassifier( ))
            ]


        , 'stacked_ensemble': [
            ('variance_check', fs.VarianceThreshold(threshold=(.995)*(1-.995))) 
            , ('clf',clf)       
            ]
    
        # , 'logistic_rfecv': [
        #     #('variance_check', fs.VarianceThreshold(threshold=(.995)*(1-.995)))
        #      ('scaler', MinMaxScaler(feature_range=(0,1)))
        #     , ('univariant_vs', univariant_sk_vs(k=20,min_occurances=1,max_out=20))
        #     , ('model_vs', _rfecv)
        #     , ('clf', LogisticRegressionCV(cv=3, scoring=custom_scorer))
        #     ]

        # , 'logistic_pca': [
        #     #('variance_check', fs.VarianceThreshold(threshold=(.995)*(1-.995)))
        #      ('scaler', MinMaxScaler(feature_range=(0,1)))
        #     , ('univariant_vs', univariant_sk_vs(k=10,min_occurances=2,max_out=20))
        #     , ("pca", PCA())
        #     , ('clf', LogisticRegressionCV(cv=3, scoring=custom_scorer))
        #     ]

        }


    #GENERATE PIPELINES FOR EACH MODEL AND FIT TO DATA 
    pipe_dic = {key:Pipeline([process for process in value]) for key, value in model_dict.items()}
    pipe_fit = {key:pipe.fit(X_train[Xs], y_train) for key, pipe in pipe_dic.items()}

    best_score = 0
    best_mod = None
    for k,v in pipe_fit.items():    
        train_score = pipe_fit[k].score(X_train[Xs],y_train)
        test_score = pipe_fit[k].score(X_test[Xs],y_test)
        print(f"{k}:{train_score}")
        print(f"{k}:{test_score}")

        if test_score > best_score:
            best_score = test_score
            best_mod = k


    # #save best model to disk 
    joblib.dump(pipe_fit[best_mod], f"{model_outdir}/{best_mod}_best_estimator.pkl")
    CFG['model']['train_params']['best_estimator'] = f"{model_outdir}/{best_mod}_best_estimator.pkl"
    with open(f"{cfg_path}", 'w') as f:
        yaml.dump(CFG, f,sort_keys=False)
        
    logger.info(f"Best Model: {best_mod} with test score of {best_score}")  
    logger.info(f"Best Model saved to {model_outdir}/{best_mod}_best_estimator.pkl")


    pipe_fit[best_mod].predict(pd.concat([X_train[Xs],X_test[Xs]])).value_counts()
























# #
# # Hyper Parameter Setup
# #
# hp_dict = {
#     'gbm_ada': {
#                       #'model_vs__max_features': [20]
#                       'clf__ccp_alpha':[0,.1,.2]
#                      , 'clf__max_depth': list(set([1,2,max_depth]))
#                      , 'clf__max_features':['auto','sqrt','log2']
#                      }
#     ,'stacked_ensemble': {'clf__rf__n_estimators': [10,n_estimators]
#                           ,'clf__gbm__n_estimators': [10,n_estimators]
#                           ,'clf__mlp__hidden_layer_sizes': [(50,25)]
#                           ,'clf__mlp__activation': ['tanh', 'relu']
#                               }
#     # , 'logistic_rfecv': {'clf__Cs': [1,2,4]}
#     # ,'logistic_pca': {'pca__n_components': [5, 15,'mle'], 'pca__whiten': [True , False]}
# }

# hp_fit ={}
# best_score = 0
# best_mod = None
# for k,v in hp_dict.items():
#     search = RandomizedSearchCV(pipe_fit[k]
#                                 , v
#                                 , n_jobs=4
#                                 , refit=True
#                                 , scoring=custom_scorer
#                                 , cv=StratifiedKFold(n_cv)
                            
#                                 )
    
#     #_X = pd.concat([X_train,X_test]) 
#     #_y = pd.concat([y_train,y_test])
#     search.fit(X_train,y_train) #pass all the data in, hp tuning occurs on a 80% split, and final model is fit on all data

#     hp_fit[k] = search.fit(X_train,y_train) 
#     print(f"{k}: Best parameter (CV score={search.best_score_}):")
#     print(f"{k}")
#     print(search.best_params_)
#     print(search.score(X_test, y_test))

#     if search.best_score_ > best_score:
#         best_est = search.best_estimator_
#         best_score = search.best_score_
#         best_mod = k 


#     del search 
#     gc.collect()


# #save best model to disk 
# joblib.dump(hp_fit[best_mod].best_estimator_, f"{model_outdir}/{best_mod}_best_estimator.pkl")
# CFG['model']['train_params']['best_estimator'] = f"{model_outdir}/{k}_best_estimator.pkl"

# with open(f"{model_outdir}/best_estimator.cfg", 'w') as f:
#     yaml.dump(CFG, f,sort_keys=False)
    



# for k,v in hp_fit.items(): 
#     print(f"{k}: {v.score(X_test, y_test)}")




# from sklearn.feature_extraction.text import TfidfVectorizer
# vectorizer = TfidfVectorizer()

# _input = pd.concat(
#     [pdf[[index_col1, text_col1]].drop_duplicates(subset=[index_col1,text_col1]).rename(
#         columns={index_col1:'index'
#                  , text_col1:'text'})
#         , pdf[[index_col2, text_col2]].drop_duplicates(subset=[index_col2,text_col2]).rename(
#             columns={index_col2:'index'
#                      , text_col2:'text'})
#                      ], ignore_index=True)

# dtm = vectorizer.fit_transform(_input['text'])
# pairwise_similarity = dtm * dtm.transpose()
# print(vectorizer.get_feature_names_out())
# print(dtm.toarray())
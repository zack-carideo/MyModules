
import logging, sys, re 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import  RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, StackingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection._univariate_selection import _BaseFilter, check_is_fitted
import sklearn.feature_selection as fs 
import numpy as np 

#logger
logger = logging.getLogger(__name__)
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

class TextSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    Use on text columns in the data
    """
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.key]
    
    def get_feature_names(self):
        return self.key
    
class CategoricSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    Use on categoric columns in the data
    """
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.key]
    
    def get_feature_names(self):
        return self.key
    
class NumberSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    Use on numeric columns in the data
    """
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[[self.key]]
    
    #return the value of the column names passed to __init__ as the feature name for the numeric variable 
    def get_feature_names(self):
        return self.key
    

def sk_univariant_vs(X,Y,k=1, max_out=1):
    selected_list = []
    selection_methods = {

        'kbest_selection_methods':[{'method':'k_best','metric':metric.__name__
                                    , 'instance': fs.SelectKBest(score_func=metric, k=k).fit(X,Y)
                                    } for metric in [fs.chi2,fs.f_classif, fs.mutual_info_classif]]

        ,'fpr_selection_methods':[{'method':'fpr','metric':metric.__name__
                                   , 'instance': fs.SelectFpr(score_func=metric, alpha=0.05).fit(X,Y)
                                   } for metric in [fs.chi2,fs.f_classif, fs.mutual_info_classif]]

        ,'fdr_selection_methods':[{'method':'fdr','metric':metric.__name__
                                   , 'instance': fs.SelectFdr(score_func=metric, alpha=0.05).fit(X,Y)
                                   } for metric in [fs.chi2,fs.f_classif, fs.mutual_info_classif]]

        ,'fwe_selection_methods':[{'method':'fwe','metric':metric.__name__
                                   , 'instance': fs.SelectFwe(score_func=metric, alpha=0.05).fit(X,Y)
                                   } for metric in [fs.chi2,fs.f_classif, fs.mutual_info_classif]]
    }

    #for each Variable selection method
        #0.return variable importance scores for each method + metric combination 
        #1. identify variables associated with each method + metric combination
        #2. identify p-vals associated with each method + metric combination 
    
    for key,list_of_dicts in selection_methods.items():
        for idx,_dict in enumerate(list_of_dicts):
            method = selection_methods[key][idx]['method']  
            metric = selection_methods[key][idx]['metric']
            try: 
                selected = selection_methods[key][idx]['instance'].get_support()
            except Exception: 
                logger.debug(f"ERROR: SUPPORT VALUES NOT AVAILABLE FOR {key} {method} {metric}")
                continue 
            
            #append boolean array of selected variables from X were selected 
            selected_list.append(selected)
            
    array_of_selections = np.vstack(selected_list) #stacking list of arrays so each list is a row in a larger r*c array
    total_hits_by_index = array_of_selections.sum(axis=0) #summing the columns to get the total number of times a variable was selected [sum_col1,sum_col2,...,sum_coln]
    ascending_indexs = np.argsort(total_hits_by_index) #get ascending indexs of the total hits by index location 
    topn_indexs = ascending_indexs[-max_out:] #get the top n indexes of the total hits by index location

    #create mask to id the n variables selected by the most variable selection methods 
    mask = [False]*X.shape[1]
    for idx in range(0,X.shape[1]):
        if idx in topn_indexs:
            mask[idx]=True
    return mask 



class univariant_sk_vs(_BaseFilter):

    """ID subset of variables with strong univarnat relationship to target"""
    def __init__(self,score_func = sk_univariant_vs, *, k=20, min_occurances=2, max_out = 10):
        super().__init__(score_func=score_func)
        self.k = k
        self.min_occurances = min_occurances
        self.max_out = max_out

    def _check_params(self, X, y):
        if not (self.k == "all" or 0 <= self.k <= X.shape[1]):
            raise ValueError(f"k should be >=0, <= n_features = {X.shape[1]}, got {self.k}. use k='all' to return all features" )


    def fit(self,X,y):
        self.mask = sk_univariant_vs(X,y,max_out=self.max_out)
        return self 
    
    def transform(self,X):
        return X[:,[idx for idx,val in enumerate(self.mask) if val]]
    
    def get_params(self,deep=False):
        return {'k':self.k, 'min_occurances':self.min_occurances}
    
    def _get_support_mask(self):
        #check_is_fitted(self)
        return np.array(self.mask)










class tfidf_document_simularity():

    def __init__(self, text_col1: str, text_col2: str):
        self.text_col1 = text_col1
        self.text_col2 = text_col2
        self.vectorizer = TfidfVectorizer()
        self.similarity = None

    def fit(self, X, y=None):

        self.vectorizer.fit(X[self.text_col1])
        return self

    def transform(self, X):
        self.similarity = self.vectorizer.transform(X[self.text_col2])
        return self

    def get_similarity(self):
        return self.similarity


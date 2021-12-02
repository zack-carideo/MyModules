from snapy import MinHash, LSH
import pandas as pd
import string 
import datetime
import numpy as np
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from text_processing import text_preprocessing as  textpp
import matplotlib.pyplot as plt
pd.set_option('max_colwidth', -1)

#start date 
start_date = datetime.datetime(2014,1,1)

#user input string to add to model *need to provide better example for how to use*
user_search = 'amazon s3 outage causes ripples'

#data specific paramters 
data_path = "C:\\Users\\zjc10\\Desktop\\Projects\\data\\news\\webhose_news\\webhose_df.pickle"
col2check = 'title'

#minhash & lsh specific parameters
#seed to enable replication 
seed = 3

#size of each overlapping text shingle to break text into prior to hashing
#set to low -> more fps
#set to high -> more fns
n_gram = 3

#specify char or term ngram 
n_gram_type = 'term'

#number of randomly sampled hash values to use for generating each texts minhash signature (larger = more accurate & slower)
permutations=1000

#hash value size to be used to generate minhash signitures from shingles (32,64, or 128 bit). 
#NOTE: should be chosen based on text length and a trade off between performance ad accuracy
hash_bits = 64

#max characters in each string to evaluate 
max_str_len = 200
max_words = 20
min_str_len = 30    

#RUN FULL SCRIPT (WRAPPER FOR minhash_class)
def run_full_lsh(col2check,data,method =1, lower=True,stopwords = None, stemmer=None):
    minhash_class = minhash_lsh(col2check)
    minhash_class.preprocess_data(data,lower=lower,stopwords=stopwords,stemmer=stemmer)
    minhash_class.build_minhash()
    minhash_class.build_lsh()
    indices_not_in_scope = set(list(data.index)) - set(minhash_class.lsh.adjacency_list().keys())
    print('{} records removed from scope becuase title was to short'.format(len(indices_not_in_scope)))
    minhash_class.find_dups_lsh(method=method)
    data['semantic_dup_info'] = minhash_class.return_dup_info(data)
    remove_me = minhash_class.dup_idxs_to_remove()
    data_nodup = data.drop(remove_me)
    data_nodup['Total_Articles'] = data_nodup['semantic_dup_info'].apply(lambda x: len(x))
    data_out = data_nodup.sort_values(by='Total_Articles',ascending=False)
    return minhash_class,data_out

class minhash_lsh():
    def __init__(self,col2check, 
                 min_str_len = min_str_len, 
                 max_str_len = max_str_len, 
                 max_words = max_words,

                 seed = seed, 
                 n_gram = n_gram , 
                 permutations = permutations , 
                 hash_bits = hash_bits ,
                ):
        
        #preprocessing
        self.min_str_len = min_str_len
        self.max_str_len = max_str_len
        self.max_words = max_words

        #minhash + lsh 
        self.col2check = col2check
        self.seed = seed 
        self.n_gram = n_gram 
        self.n_gram_type = n_gram_type
        self.permutations = permutations
        self.hash_bits = hash_bits
        self.docs = None 
        self.labels = None 
        self.minhash = None
        self.lsh = None
        
        #dup info 
        self.dup_dict = None

    def preprocess_data(self, data_in,lower=False,stopwords = None, stemmer=None):
        
        #dont alter data, make copy(this way 'title' is unimpacted in the parent df)
        data = data_in.copy()
        
        if stopwords: 
            #clean text 
            data[self.col2check] = data[self.col2check].apply(lambda title: textpp.clean_sentence(title,
                                                                                        lower=lower,
                                                                                        stopwords=stopwords, 
                                                                                        stemmer = stemmer,
                                                                                       ))
        else:
            data[self.col2check] = data[self.col2check].apply(lambda title: textpp.clean_sentence(title,lower=lower))
        
        Data = data[~(data[self.col2check].isna()) & (data[self.col2check].str.len()>self.min_str_len)]

        #truncate text to ensure consistent string length comparisons 
        Data[self.col2check] = Data[self.col2check].apply(lambda x: ' '.join(x.split()[:self.max_words])[:self.max_str_len])

        #create label set to use in lsh model 
        self.labels = [i for i in Data.index]
        self.docs = Data[self.col2check].copy()
        
        

    #build minhash signitures 
    def build_minhash(self):
        self.minhash = MinHash(self.docs, n_gram=self.n_gram, permutations=self.permutations,n_gram_type=self.n_gram_type,
                               hash_bits = self.hash_bits, seed=self.seed)

    #get lsh model to query 
    def build_lsh(self):
        self.lsh = LSH(self.minhash, self.labels, no_of_bands=self.permutations/2)

    #add new text to lsh model (single text)
    #create minhash sigs for new text
    #update lsh model with new hash sigs
    def update_lsh(self, new_text, new_label):
        new_minhash = MinHash([new_text] , n_gram = self.n_gram , permutations = self.permutations,n_gram_type=self.n_gram_type,
                              hash_bits = self.hash_bits, seed = self.seed)
        self.lsh.update(new_minhash, [new_label])

    #query updated lsh model for new string input 
    def query_something_new(self,label,min_sim = .4):
        return self.lsh.query(label,min_jaccard=min_sim)
      
    #iterate over 1st and second order relationships to find full 'related article list to use as dups'
    def find_dups_lsh(self,method=2):
        #using lsh to pull back similar titiles (pulling all primary and secondary relationships as dups) 
        used_docs = []
        self.dup_dict = {}
        
        if method ==1:
            for item in self.lsh.adjacency_list().items():
                vals = []
                if item[0] not in used_docs:
                    used_docs.append(item[0])
                    for val in item[1]:
                        if val not in used_docs:
                            used_docs.append(val)
                            vals.append(val)
                    self.dup_dict[item[0]] = vals 
        else:

            #loop through all keys in adjaency list(essentially loop through docs)
            for key in self.lsh.adjacency_list().keys():
                vals = []

                #make sure if a key is used, its not included in any other itemset 
                if key not in used_docs:
                    used_docs.append(key)

                    #query the key (primary article) to find all articles related to that key
                    simdocs = self.query_something_new(key,min_sim = 0)

                    #for each article related to that key repeat the query process to identify all articles 
                    #related to that key!(second order relationships)
                    simdocs2 = []
                    for val in simdocs:
                        simdocs2.append(val)
                        simdocs2 = simdocs2+self.query_something_new(val,min_sim = 0)

                    #get unique list of articles associated with primary article 
                    simdocs_out = list(set(simdocs2))
                    for val in simdocs_out: 
                        if val not in used_docs:
                            used_docs.append(val)
                            vals.append(val)

                    #update dup_dict key with all primary and secondary relationships associated with parent
                    self.dup_dict[key] = vals 

    #get list of all dups associated with 
    def return_dup_info(self,Data):
        semantic_dup_info_list = []
     
        for idx in Data.index:
            semantic_dup_idx = self.dup_dict.get(idx,[])

            if semantic_dup_idx !=[]:
                semantic_dup_info_list.append(Data.loc[semantic_dup_idx,['title']].to_dict("records"))
            
            else:
                semantic_dup_info_list.append({})

        return semantic_dup_info_list

    def dup_idxs_to_remove(self):
        return [item for sublist in [self.dup_dict[key] for key in self.dup_dict.keys()] for item in sublist]


if __name__ == "__main__":
    #constants for use in sample script 
    PUNC = set(string.punctuation)
    STOPWORDS = stopwords.words('english')
    STEMMER = SnowballStemmer('english')

    d = pd.read_pickle(data_path).reset_index()
    data =d.sample(1000)
    minhash_class,data_out = run_full_lsh('title',data)
    data_out.sort_values(by='Total_Articles',ascending=False,inplace=True)
    print('deduped dataset contains {} records'.format(data_out.shape[0]))
    print('sample of titles associated with duplicate entries')
    print(data_out['semantic_dup_info'].head(5))

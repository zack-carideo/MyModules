#https://towardsdatascience.com/master-semantic-search-at-scale-index-millions-of-documents-with-lightning-fast-inference-times-fa395e4efd88
#https://www.kaggle.com/code/nandhuelan/semantic-search

import faiss , time
import numpy as np 
import pandas as pd
import sentence_transformers
from sentence_transformers import SentenceTransformer,  CrossEncoder 

class faiss_index:
    
    def __init__(self
                 , data:pd.DataFrame
                 , model: sentence_transformers.SentenceTransformer
                 , embed_dim: int 
                 , text_col:str = None
                 , id_col:str = None 
                 , index_outpath: str = None
                 , cross_encoder_model_name: str = None
                 ):
        """_summary_

        Args:
            data (pd.DataFrame): dataframe containing text we want to index, and the doc2text mapping columns 
            model (sentence_transformers.SentenceTransformer): the sentence transformer to use in generating embeddings. The transformer should be one geared for the task assocaited with the index (information retrieval , semantic search, q&a, etc....)
            embed_dim (int): dimensions (ex. 768) of the transformer model being used
            text_col (str): the name of the text field in the dataframe to model  
            id_col (str, optional): the column in the dataframe to use as a named index on the output FAISS index 
            index_outpath (str, optional): location to save '.index' file to use for post trained index queries
            cross_encoder_model_name (str, optional): _description_. Defaults to None.
        """
        self.data=data
        self.model=model
        self._text_col = text_col
        self._id_col = id_col
        self._embed_dim= embed_dim 
        self._index_outpath = index_outpath
        self._cross_encoder_model_name = cross_encoder_model_name
        
        self._ids = None
        if cross_encoder_model_name is not None:  
            self._ce = CrossEncoder(cross_encoder_model_name)
        else:
            self._ce = None 
            
    def index(self):
        """create initial index 
        """
        #encode data passages using trained bi-encoder  
        encoded_data = self.model.encode(
            self.data[self._text_col].values.tolist())
        encoded_data = np.asarray(encoded_data.astype('float32'))
        self._ids = [(idx,v) for idx,v in enumerate(self.data[self._id_col])]
        
        #note only indexflatIP and indexFlatL2 gaurentee exact results (no clustering)
        self.index = faiss.IndexIDMap(
            faiss.IndexFlatIP(self._embed_dim))
        
        #add contextual index to faiss 
        self.index.add_with_ids(encoded_data , [__id[0] for __id in self._ids])
        
        #write faiss index to disk 
        if self._index_outpath is not None: 
            assert self._index_outpath.split('.')[-1] =='index', 'you must save index to .index file extension'
            faiss.write_index(self.index, self._index_outpath)
        
        
    def fetch(self,idx,sim_score):
        info = self.data.iloc[idx]
        
        meta_dict = {}
        #meta_dict[f"{self._text_col}_{sim_score}"] = info[self._text_col]
        meta_dict[f"{info[self._id_col]}"] = {'text':info[self._text_col]
                                              ,'score': sim_score
                                              }
               
        print(meta_dict)
        return meta_dict

    def cross_encode_fetched(self,query, biencoder_results):
        
        assert self._ce is not None, 'you must have a defined and compiled cross encoder model to use this function'
        #format data for input to cross encoder
        ce_in = [[query, _d[next(iter(_d))]['text']] for _d in biencoder_results  ]

        #re-rank embeddings from bi-encoder using cross encoder
        cross_scores = self._ce.predict(ce_in)
        
        #return ranked list of cross encoder scores 

        return [(cross_scores[hit] ,"\t{}".format(biencoder_results[hit][next(iter(biencoder_results[hit]))]['text'].replace("\n", " "))) 
                for hit in np.argsort(np.array(cross_scores))[::-1]]
    
    def search(self,query, top_k, refine_with_crossencoder=False):
        t=time.time()
        query_vector = self.model.encode([query])
        top_k = self.index.search(query_vector, top_k)
        print('>>>> Results in Total Time: {}'.format(time.time()-t))
        print(top_k)
        
        top_k_ids = top_k[1].tolist()[0]
        top_k_sims = top_k[0].tolist()[0]
        top_k_ids = list(np.unique(top_k_ids))
        print(top_k_ids)
        print(top_k_sims)
       
        results =  [self.fetch(id, top_k_sims[idx]) for idx,id in enumerate(top_k_ids)]
        #results =  [f"{top_k_sims[idx]}_{self.fetch(idx)}" for idx in top_k_ids]
        
        if refine_with_crossencoder==False: 
            return results, top_k_ids
        else:
            return self.cross_encode_fetched(  query
                                      , results)

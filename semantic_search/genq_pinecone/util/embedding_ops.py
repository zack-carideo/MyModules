import torch, gc
from tqdm.auto import tqdm
import pathlib
from pathlib import Path 
from typing import List ,  Tuple
import pandas as pd 

from sentence_transformers import InputExample, datasets, models, SentenceTransformer
from tqdm.auto import tqdm

def create_batch(iterable, batch_size=100):
    length = len(iterable)
    for ndx in range(0, length, batch_size):
        yield iterable[ndx: (ndx + batch_size)]
        
        
class query_ops:

    def __init__(
        self
        , tokenizer
        , model 
        , out_dir 
        , n_queries_per_passage = 3
        , train_batch_size = 5
        , save_batch_size = 1000
        
        #tokenizer params
        , return_tensors='pt'
        , padding='max_length'
        , return_overflowing_tokens = True
        , max_seq_len = 300
        , truncation = True
        , stride = 50

    ):
        """_summary_

        Args:
            tokenizer (_type_): _description_
            model (_type_): _description_
            out_dir (_type_): _description_
            n_queries_per_passage (int, optional): _description_. Defaults to 3.
            batch_size (int, optional): _description_. Defaults to 50.
            return_tensors (str, optional): _description_. Defaults to 'pt'.
            padding (str, optional): _description_. Defaults to 'max_length'.
            return_overflowing_tokens (bool, optional): if seq exceeds max length, and return_overflowing_tokens=True, a new seq with the truncated text will  be generated instead of truncating text > max_seq_length. Defaults to True.
            max_seq_len (int, optional): max sequence length to use in tokenization. Defaults to 300.
            truncation (bool, optional): truncate texts>max_len. Defaults to True.
            stride (int, optional): Total overlapping tokens to use in sliding window. Defaults to 50.
        """
        
        self._out_dir =  out_dir
        self._model = model 
        self._tokenizer = tokenizer 
        self._n_queries_per_passage = n_queries_per_passage
        self._train_batch_size = train_batch_size
        self._save_batch_size = save_batch_size 
        
        self._return_tensors = return_tensors
        self._padding = padding
        self._return_overflowing_tokens = return_overflowing_tokens 
        self._max_seq_length = max_seq_len
        self._truncation = truncation
        self._stride = stride
        
        self._query_passage_outpaths = []
        self._passage2chunk_map = None
        
    def load_query_passage_pairs(self):
        """load query passage pairs generated from t5 
        Args:
            query_passage_tsv_dir (pathlib.Path): path to decoded t5 output of query, passage pairs (this is a directory)
        Returns:
            _type_: _description_
        """
        
        assert Path(self._out_dir).suffix == '.tsv' , 'the query passage pairs should be saved and loaded from a tsv file'
        return [str(path) for path in Path(self._out_dir).glob('*.tsv')]


    def gen_queries_from_passages(self
                                , list_of_texts: List
                                ):
        
        """takes a list of passages and executes the below steps 
        1. encode all passages and return pytorch tensors
        2. generate output tokens from input embeddings (aka generate n queries linked to  each passage) 
        3. decode output  tokens into human readable text 
        4. append decoded queries to associated input passage 
        5. save all query , passage pairs to disk 
        
        Args:
            list_of_texts (List): list of passages you with to generate n queries for each   list_of_texts = [(doc #, text), (doc #, text), ... , ]
        """
        
        assert self._out_dir is not None, 'please specificy output directory to save embedded text'
        
        #containers for outputs
        scored_df_list = []
        file_count = 0


        # set to no_grad as we don't need to calculate gradients for back prop
        with torch.no_grad():
            
            # loop through each passage individually
            for p in tqdm(create_batch(list_of_texts,self._train_batch_size)):
    
                
                torch.cuda.empty_cache()
                gc.collect()
        
                #extract docs
                docs = [p_[0] for p_ in p ]
                
                #clean tabs 
                p = [p_[1].replace('\t', ' ') for p_ in p]
                
                # encode input tokens and return as pytorch tennsors
                # ENCODE THE TOKENIZED PASSAGE THAT YOU WANT TO GENERATE QUIERIES FOR 
                # DONT FORGET TO PUSH THE DATA TO THE GPU TO ENABLE GPU BASED PROCESSING  (.to('cuda'))
                input_ids = self._tokenizer(p
                                            , return_overflowing_tokens = self._return_overflowing_tokens                                            #, return_tensors=self._return_tensors
                                            , truncation= True 
                                            , padding = True
                                            , stride = self._stride
                                            , max_length = self._max_seq_length
                                            )
                
                # create  mapping of passages to chunks (for audit policy tracking)
                # decoding because the tokenizer generated a sliding window wwith span 
                # that we must keep track of to keep doc to chunk alignment 
                # passage2chunk_map =[{'text':a,'doc':docs[b]} for a,b in 
                #     zip(*[self._tokenizer.batch_decode(input_ids.input_ids),
                #         input_ids['overflow_to_sample_mapping']])
                #     ]
                
                passage2chunk_map = []
                _lastdoc = None
                idxx = 0 
                for idx,tup in enumerate(zip(*[self._tokenizer.batch_decode(input_ids.input_ids),input_ids['overflow_to_sample_mapping']])):
                    
                    #inter document chunk index (must be a better way!)
                    if idx==0:
                        _lastdoc = tup[1]
                    else:
                        #if current doc == last , add 1 to doc chunk iderator
                        if tup[1]==_lastdoc:
                            idxx = idxx+1
                        else:
                            idxx=0
                            _lastdoc = tup[1]  
                            
                    #create map of each text, its doc index, and the respect doc chunk
                    passage2chunk_map.append({'text':tup[0],'doc':docs[tup[1]] , 'chunk':idxx})
                
    
        
                
                #generate queries 
                outputs = self._model.generate(
                    
                    # Indices of input sequence tokens in the vocabulary`
                    # is passing the input sequence tokens to the model for query generation. The
                    # `input_ids` parameter is a tensor containing the indices of the input tokens in the
                    # vocabulary. It is used by the model to understand the input and generate the
                    # corresponding queries.
                    input_ids = torch.Tensor(input_ids.input_ids).type(torch.int64).to('cuda')
                    
                    # The `max_length=64` parameter in the `model.generate()` function is setting the
                    # maximum length of the generated query. It ensures that the generated query will not
                    # exceed 64 tokens.
                    , max_length = 64
                    
                    # The `do_sample=True` parameter in the `model.generate()` function is used to enable
                    # sampling during the generation of queries. When `do_sample=True`, the model will
                    # randomly sample from the top-k most likely next tokens, where k is determined by the
                    # `top_p` parameter. This allows for more diverse and creative query generation.
                    , do_sample = True
                    
                    # The `top_p=0.95` parameter in the `model.generate()` function is used to control the
                    # diversity of the generated queries.
                    , top_p = .95

                    # The `num_return_sequences=3` parameter in the `model.generate()` function is used to
                    # specify the number of different query sequences to generate for each input passage.
                    # In this case, it is set to 3, which means that for each input passage, the model
                    # will generate 3 different query sequences.
                    , num_return_sequences = self._n_queries_per_passage         
                                            ).to('cuda')
                

                #update  query passage map to hold text, doc, queries (bring back to cpu)
                #use batch_size = number queries generated per each chunk, this ensures your decoding the right quereis for right doc
                for idx,ec_query in enumerate(create_batch(outputs,self._n_queries_per_passage)):
                    passage2chunk_map[idx].update({'ec_query_ids':ec_query.to('cpu')})
                    passage2chunk_map[idx].update({'ec_query_txt':[self._tokenizer.decode(q_, skip_special_tokens=True) 
                                                                for q_ in ec_query.to('cpu')]})
            
            
                    #write aligned  query, passages to df 
                    scored_df_list.append(pd.DataFrame.from_dict({k:v for k,v in passage2chunk_map[idx].items() if k!='ec_query_ids'}
                                        ,orient='columns')
                    )
                    
                
                if sum([x.shape[0] for x in scored_df_list])> self._save_batch_size:
            
                    out_p = f'{self._out_dir}/pairs_{file_count}.pq'
                    pd.concat(scored_df_list).to_parquet(out_p)
                    self._query_passage_outpaths.append(out_p)
                    
                    file_count+=1
                    scored_df_list = []
            
            #clean up any reamaining 
            if len(scored_df_list)>0:
                out_p = f'{self._out_dir}\pairs_{file_count}.pq'
                pd.concat(scored_df_list).to_parquet(out_p)
                self._query_passage_outpaths.append(out_p)

        #self._passage2chunk_map = passage2chunk_map
        
        return [v for v in self._query_passage_outpaths]



    def create_training_data(self
                             ,query_passage_outpaths: list = None
                             , query_col = 'ec_query_txt'
                             , passage_col = 'text'
                             , passage_idx_col = 'doc'
                             , passage_chunk_idx_col = 'chunk'
                             ):
        
        """create a sentence_transformer training dataset of query,passage pairs to use in fine tunning a LLM  
        
        query_passage_outpaths: list of texts , each text has a query and passage delimited by \t 
        

        """
        
        if query_passage_outpaths is None: 
            query_passage_outpaths = self._query_passage_outpaths
        
        
        #create df, and use original index as doc chunk idx 
        #df['chunk_idx'] = df.groupby(['doc']).cumcount()+1
        df = pd.concat([pd.read_parquet(p) for p in self._query_passage_outpaths]
                       ).reset_index(drop=True)#.rename(columns={'index':passage_chunk_idx_col})
        
        #create single index col 
        df['_index'] = df[passage_idx_col].astype(str) + '_' + df[passage_chunk_idx_col].astype(str)
        #update passage to include location informatin within text
        df['passage'] = df['_index'] +":"+df[passage_col]
        
        
        pairs = [(_d[query_col] 
                          ,_d['passage']) 
                         for idx,_d in df.iterrows()]
        
        return df , pairs 


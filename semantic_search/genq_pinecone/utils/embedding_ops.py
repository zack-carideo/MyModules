import torch
from tqdm.auto import tqdm
import pathlib
from pathlib import Path 
from typing import List ,  Tuple

from sentence_transformers import InputExample, datasets, models, SentenceTransformer
from tqdm.auto import tqdm

class query_ops:
    
    def __init__(
        self
        , tokenizer
        , model 
        , out_dir 
        , n_queries_per_passage = 3
        , batch_size = 50

    ):
        
        self._out_dir =  out_dir
        self._model = model 
        self._tokenizer = tokenizer 
        self._n_queries_per_passage = n_queries_per_passage
        self._batch_size = batch_size 
        
        self._query_passage_outpaths = []
        
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
            list_of_texts (List): list of passages you with to generate n queries for each 
        """
        
        assert self._out_dir is not None, 'please specificy output directory to save embedded text'
        
        #containers for outputs
        pairs = []
        file_count = 0

        # set to no_grad as we don't need to calculate gradients for back prop
        with torch.no_grad():
            
            # loop through each passage individually
            for p in tqdm(list_of_texts):
                
                p = p.replace('\t', ' ')
                
                # encode input tokens and return as pytorch tennsors
                # ENCODE THE TOKENIZED PASSAGE THAT YOU WANT TO GENERATE QUIERIES FOR 
                # DONT FORGET TO PUSH THE DATA TO THE GPU TO ENABLE GPU BASED PROCESSING  (.to('cuda'))
                input_ids = self._tokenizer.encode(p, return_tensors='pt' , truncation=True).to('cuda')
                
                # generate output tokens from input passage
                # OUTPUT TOKENS = QUERIES , so we are passing in the long passage and the model is generating n  queries based on the passage
                # our queries will have a max length of 64 tokens and the variation across queries will be low (p=.95)
                # THIS IS OCCURING 1 text at a time, THERE HAS TO BE AN EASY WAY TO EXECUTE THIS IN BATCHES!?!? 
                outputs = self._model.generate(
                    
                    # Indices of input sequence tokens in the vocabulary`
                    # is passing the input sequence tokens to the model for query generation. The
                    # `input_ids` parameter is a tensor containing the indices of the input tokens in the
                    # vocabulary. It is used by the model to understand the input and generate the
                    # corresponding queries.
                    input_ids=input_ids, 
                    
                    # The `max_length=64` parameter in the `model.generate()` function is setting the
                    # maximum length of the generated query. It ensures that the generated query will not
                    # exceed 64 tokens.
                    max_length=64,
                    
                    # The `do_sample=True` parameter in the `model.generate()` function is used to enable
                    # sampling during the generation of queries. When `do_sample=True`, the model will
                    # randomly sample from the top-k most likely next tokens, where k is determined by the
                    # `top_p` parameter. This allows for more diverse and creative query generation.
                    do_sample=True,
                    
                    # The `top_p=0.95` parameter in the `model.generate()` function is used to control the
                    # diversity of the generated queries.
                    top_p=0.95,
                    
                    # The `num_return_sequences=3` parameter in the `model.generate()` function is used to
                    # specify the number of different query sequences to generate for each input passage.
                    # In this case, it is set to 3, which means that for each input passage, the model
                    # will generate 3 different query sequences.
                    num_return_sequences=self._n_queries_per_passage
                )
   
                #loop over the n encoded queries generated for passage(i) and decode them into 
                #human readable form 
                for output in outputs:
                    
                    query = self._tokenizer.decode(output, skip_special_tokens=True)
    
                    # append (query, passage) pair to pairs list, separate by \t
                    pairs.append(query.replace('\t', ' ')+'\t'+p)
                
                # once we have 1024 pairs write to file
                if len(pairs) > self._batch_size:
                    with open(f'{self._out_dir}/pairs_{file_count}.tsv', 'w', encoding='utf-8') as fp:
                        fp.write('\n'.join(pairs))
                
                    file_count += 1
                    pairs = []
                    self._query_passage_outpaths.append(f'{self._out_dir}/pairs_{file_count}.tsv')

        if pairs is not None:
            # save the final, smaller than 1024 batch
            with open(f'{self._out_dir}/pairs_{file_count}.tsv', 'w', encoding='utf-8') as fp:
                fp.write('\n'.join(pairs))

            self._query_passage_outpaths.append(f'{self._out_dir}/pairs_{file_count}.tsv')

        return [v for v in self._query_passage_outpaths]

    def create_training_data(self
                             , query_passage_outpaths: List = None
                             ):
        
        """create a sentence_transformer training dataset of query,passage pairs to use in fine tunning a LLM  
        
        query_passage_outpaths: list of texts , each text has a query and passage delimited by \t 
        

        """
        
        if query_passage_outpaths is None: 
            query_passage_outpaths = self._query_passage_outpaths
            
        pairs = []
        for path in tqdm(query_passage_outpaths):
            with open(path, 'r', encoding='utf-8') as fp:
                lines = fp.read().split('\n')
                for line in lines:
                    if '\t' not in line:
                        continue
                    else:
                        q, p = line.split('\t')
                        pairs.append(
                            (q,p)
                            )
        
        return pairs 
     


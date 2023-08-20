import sys ,os ,logging, copy,  torch, pysbd, transformers, random, faiss
import pandas as pd
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from collections import OrderedDict
from typing import List, Tuple, Dict
from itertools import chain 

#set up basic logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
logger =  logging.getLogger(__name__)



def create_batch(iterable, batch_size=100):
    length = len(iterable)
    for ndx in range(0, length, batch_size):
        yield iterable[ndx: (ndx + batch_size)]
        
def flatten_dict(dd, separator='_', prefix=''):
    stack = [(dd, prefix)]
    flat_dict = {}

    while stack:
        cur_dict, cur_prefix = stack.pop()
        for key, val in cur_dict.items():
            new_key = str(cur_prefix) + separator + str(key) if cur_prefix else str(key)
            if isinstance(val, dict):
                stack.append((val, new_key))
            else:
                flat_dict[new_key] = val

    return flat_dict


#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class sent_embedder:
    
    """class to handle the conversion of text to sentence embeddings 
    """
    
    def __init__(
        self
        , language = 'en'
        , clean = True
        , LLM_MODEL_NAME = 'sentence-transformers/bert-base-nli-mean-tokens'
        , device =  'cuda'

    ):
        #set device 
        self._device = device 
        
        #initalize segmenter with class so you only need one instance for all text 
        self._parser = self.parser = pysbd.Segmenter(language=language
                                                    , clean = clean)

        #initalize tokenizer 
        self._tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
        
        #intialize model 
        self._model = AutoModel.from_pretrained(LLM_MODEL_NAME)   
        self._model.to(device)
        
        #index of doc_sent information for retrieval after sent embeds
        self._doc_sent_index_map = None 
        self._sent_dic = None 
        

    def index_text(self, text_list: List)-> Dict:
        """creating an indexed representation of the sentences in the `text_list`.
        Args:
            text_list (_type_): _description_
        """
        return {
                idx:{ 
                    idxx:sent for idxx,sent in enumerate(self._parser.segment(txt))
                    } for idx,txt in enumerate(text_list)
                }
        
    def parse_txt(self,text_list: List) -> Dict :
        
        """
        index text, and create flat index of doc#_sent# and 
        save to class instance for use in querying embeddings
        Args:
            text_list (List): list of documents that have not been tokenized or parsed 
        Returns:
            Dict: dictonary where keys are doc_sent indcies and values contain the sentence text  
        """
        
        #index text (doc(i): {sent1:text, sent2:text, ...,sentn:text})
        index_text_ = self.index_text(text_list)
        
        #flatten indexed text into ordered dict with key = doc_sent
        sent_dic = flatten_dict(index_text_, separator='_', prefix='')
        
        #store index to class for use after generating embeddings
        self._doc_sent_index_map = list(sent_dic.keys())
        self._sent_dic = sent_dic 
        
        return   sent_dic

    
    def tokenize_text_dict(self, sentence_dict: dict , batch_size:int = 10)-> List[torch.TensorType]:
        """use huggy faces auto tokenizer to convert text into input ids in the manner consistent with the embedding model being used 
        Args:
            sentence_dict (dict): dictonary of sentence tokenized texts with keys holding doc_sent index for tracing 
            batch_size (int, optional): total number of sentences to generate input tokens , type tokens, 
                                        and atten mask tokens in each batch. each batch is passed to model for generating embeddings. Defaults to 10.
        Returns:
            List[torch.Tensor]: list of tensors, where each element in list is of shape (batch_size, embedding_dim)
                                and total elements in list =  len(sentence_dict) / batch_size, and 
        """
        
        assert sentence_dict is not None , 'sentence dict must be populated before tokenizing can occur'
        
        #generate batches of transformer batch encodings to pass to model         
        return [self._tokenizer(list(x_chunk)
                                        , padding=True
                                        , truncation=True
                                        , return_tensors='pt'
                                        , max_length= 500
                                        , add_special_tokens=True
                                        ) 
                for x_chunk in create_batch(list(sentence_dict.values()),batch_size)]

    def gen_sent_embeds(self
                        , encoded_input: List[transformers.tokenization_utils_base.BatchEncoding]
                        ) -> List[List[torch.TensorType]]:
        
        
        """Standardized process to generate sentence embeddings without use of huggyface

        Args:
            encoded_input (List[transformers.tokenization_utils_base.BatchEncoding]): _description_

        Returns:
            List[List[torch.TensorType]]: _description_
        """
        #extract seq embeddings and conduct mean pooling to get sent level info  
        # Perform pooling.
        with torch.no_grad():

            # sent_embed_batches = [mean_pooling(self._model(**ec.to('cuda')) , ec['attention_mask'])
            #                              for ec in encoded_input]
            sent_embed_batches = []
            for ec in encoded_input:
                
                #grab input sequence vocab ids and send to gpu
                ids = ec.input_ids.to(self._device)
                
                #grab atten mask and send to gpu
                attnmsk = ec['attention_mask'].to(self._device)
                
                #generate sent embeddings
                se_ = self._model(ids, attention_mask = attnmsk) 
                
                #weighted mean pool embeddings using attention mask  
                sent_embed_batches.append(mean_pooling(se_, attnmsk))
            
        return sent_embed_batches

    def e2e(self, text_list: List) -> List[torch.Tensor]:
        """function to execute e2e tokenization and embedding creation from a list of texts

        Args:
            text_list (List): list of texts of any length , where each element in list represents a holistic text
        Returns:
            List[torch.Tensor]: returns list of sentence embeddings. where by each doc is tokenized into sentences, the word embeddings in each sentence are pooled into sentence embeddings 
        """
        
        logger.info('parsing documents into sentence indexed dictionary')
        sentence_dic = self.parse_txt(text_list)
        
        logger.info('tokenizing doc_sent dict to preprare for batched conversion to sent embeddings')
        encoded_input = self.tokenize_text_dict(sentence_dic)
        
        #return encoded_input
        logger.info('generating sentence emebddings')
        return self.gen_sent_embeds(encoded_input)
        

import sys,os,logging, torch, faiss, pysbd
import pandas as pd
from pathlib import Path
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch 

#set up basic logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
logger =  logging.getLogger(__name__)

#config path 
root_ = os.path.abspath("")
cfg_path = Path(root_) / "config.yaml"

#custom imports
sys.path.append(root_)
from utils.misc import LoadCFG, seed_all
from utils.data import load_data
from utils.embed_ops import sent_embedder
from utils.index_ops import index_embeddings , query_index, add_to_index

#set seed 
SEED = 42
seed_all(SEED)

#load cfg params
cfg = LoadCFG(cfg_path, base_dir = root_).load()
DATA_PATH = cfg.data.input.data_path
SAVE_DIR = Path(cfg.data.output.data_save_dir)
MODEL_SAVE_DIR = Path(cfg.model.model_save_dir)
LLM_MODEL_NAME = cfg.model.llm_model_name 
BATCH_SIZE = cfg.model.batch_size
NSAMPS = cfg.model.n_samps
TEXT_COL = cfg.model.text_col
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#inference queries
queries = cfg.inference.queries 

#get sample of non null complaints 
docs = load_data(DATA_PATH).query(f"{TEXT_COL}.notnull()", engine='python'
                                      ).sample(20)[TEXT_COL].values

#tokenize docs to sents and index them 
processer = sent_embedder(
         language = 'en'
        , clean = True
        , LLM_MODEL_NAME = 'sentence-transformers/bert-base-nli-mean-tokens'
        )

#gen sent_embeds and create index
#tst = list(chain(*sentence_embeddings))
sentence_embeddings = [se.to('cpu') for se in processer.e2e(docs)]
se_v = torch.vstack(sentence_embeddings)
index_sent_dic = processer._sent_dic
index = index_embeddings(se_v)


#query index
res_df = [] 
for query in queries: 
    se_q = torch.vstack([se.to('cpu') for se in processer.e2e([query])])
    res_df.append(query_index(processer, index, index_sent_dic, query).head(3))
    
    #this isnt working (I CANT FIGURE OUT HOW TO ADD TO AN INDEX)
    index = add_to_index(index, se_q)
    print(index.__sizeof__())
print(pd.concat(res_df))

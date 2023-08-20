import sys,os,logging
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
from utils.embedding_ops import query_ops
from utils.model_ops import build_model 

#set seed 
SEED = 42
seed_all(SEED)

#load cfg params
cfg = LoadCFG(cfg_path, base_dir = root_).load()
DATA_PATH = cfg.data.input.data_path
SAVE_DIR = Path(cfg.data.output.data_save_dir)
MODEL_SAVE_DIR = Path(cfg.model.model_save_dir)
LLM_MODEL_NAME = cfg.model.llm_model_name 
BI_ENCODER_MODEL_NAME = cfg.model.bi_encoder_model_name
EPOCHS = cfg.model.n_epochs
BATCH_SIZE = cfg.model.batch_size
NSAMPS = cfg.model.n_samps

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#create output folder if it doesnt exist
if not SAVE_DIR.is_dir():
    assert (not SAVE_DIR.is_file()), f'a directory to save outputs must be passed, you passed a full file path: {save_dir}'
    if not SAVE_DIR.parent.is_dir(): 
        os.mkdir(str(SAVE_DIR.parent))
        os.mkdir(str(SAVE_DIR))
    else:
        os.mkdir(str(SAVE_DIR))
    logger.info(f"new output directory created:{SAVE_DIR}")

if not MODEL_SAVE_DIR.is_dir():
    assert SAVE_DIR.parent.is_dir(), f'parent directory: {SAVE_DIR} does not exist'
    os.mkdir(str(MODEL_SAVE_DIR))

#load data (directly from hf, to load from file please reference load_data() definition)
#Note: we are loading a squad dataset made for NLI type operations
logging.info('loading data from huggyface')
df = load_data( load_from_directory=False , hf_dataset_name = 'squad' , split ='train') 


# we want to emulate the scenario in which we do not have queries. 
# We will remove all but the 'context' data to do that. (aka all that is passed into framework is list of text)
logging.info('extracting text passages to generate queries for')
passages = list(set(df['context']))[:NSAMPS]
print(len(passages))

#Generate queries from passages to use to generate syntheic key,value pairs of (Questions, Awnsers) 
#initalize tokenizer and use t5 as query generation model used to generate queries associated with input passages 
#Note: we are using a t5 model fine tuned for query generation as part of the BeIR Project 
logger.info('creating tokenizer and model to use in bi-encoder')
tokenizer  = T5Tokenizer.from_pretrained(LLM_MODEL_NAME, legacy=False, truncation=True, max_len=500) 

logger.info('creating model to use in bi-encoder')
model = T5ForConditionalGeneration.from_pretrained(LLM_MODEL_NAME)

#call eval() to force / ensure model is running in 'INFERENCE MODE' and not 'TRAINING' mode
logger.info('forcing model into eval mode')
model.eval()
model = model.to(device)
tokenizer = tokenizer
print(device)

#initalize class to generate queries from passages
logger.info('initalize embedding querier')
queryer = query_ops(
     tokenizer
    , model 
    , SAVE_DIR
    , n_queries_per_passage = 3
    , batch_size = 3
    )

#generate query,passage key value pairs , save to disk , return paths 
logger.info('generating query, passage key value pairs')
query_passage_outpaths = queryer.gen_queries_from_passages(passages)

#create sentence_transformers comptable training dataset using InputExample() method from transformers
logger.info('creating training data for bi-encoder fine tuning')
pairs = queryer.create_training_data( query_passage_outpaths)

#create object to handle loading of InputExample() instances in batches of 50 
logger.info('creating loader to handle loading batches of data for model training')
#loader = queryer.gen_model_data_loader(pairs, batch_size = 2)

#build and train the bi-encoder to be used for asymetric search 
logger.info('building model')
model = build_model(pairs
                    , BI_ENCODER_MODEL_NAME
                    , str(MODEL_SAVE_DIR / 'fine_tuned_biencoder')
                    , epochs=EPOCHS)
    
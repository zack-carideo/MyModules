import sys,os,logging, gc
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, T5Tokenizer,T5TokenizerFast, T5ForConditionalGeneration
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
from util.misc import LoadCFG, seed_all
from util.data import load_data
from util.embedding_ops import query_ops
from util.model_ops import build_model 
from util.index_ops import ScalableSemanticSearch

#set seed 
SEED = 42
seed_all(SEED)

#load cfg params
cfg = LoadCFG(cfg_path, base_dir = root_).load()
DATA_PATH = cfg.data.input.data_path
SAVE_DIR = Path(cfg.data.output.data_save_dir)
MODEL_SAVE_DIR = Path(cfg.model.model_save_dir)

NSAMPS = cfg.model.n_samps
TOK_BATCH_SIZE = cfg.model.tokenizer.batch_size
BI_ENCODER_MODEL_NAME = cfg.model.bi_encoder.model_name
EPOCHS = cfg.model.n_epochs
BI_ENCODER_BATCH_SIZE =  cfg.model.bi_encoder.batch_size


#device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#tokenizer setup
return_tensors = cfg.model.tokenizer.return_tensors
padding =  cfg.model.tokenizer.padding
return_overflow_tokens= cfg.model.tokenizer.return_overflow_tokens
max_seq_len = cfg.model.tokenizer.max_seq_len
truncation = cfg.model.tokenizer.truncation 
stride = cfg.model.tokenizer.stride 

#query generator setup
GENQ_MODEL_NAME = cfg.model.query_gen.model_name 
N_QUERIES_PER_PASSAGE =  cfg.model.query_gen.n_queries_per_passage 

#clean up gpu 
torch.cuda.empty_cache()
gc.collect()
logger.info(torch.cuda.memory_summary(device='cuda', abbreviated=True))

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

##
##
##

#load data (directly from hf, to load from file please reference load_data() definition)
#Note: we are loading a squad dataset made for NLI type operations
logging.info('loading data from huggyface')
df = load_data( load_from_directory=False 
               , hf_dataset_name = 'squad' 
               , split ='train') 


# we want to emulate the scenario in which we do not have queries. 
# We will remove all but the 'context' data to do that. (aka all that is passed into framework is list of text)
logging.info('extracting text passages to generate queries for')
passages = list(set(df['context']))[:NSAMPS]
print(len(passages))

#Generate queries from passages to use to generate syntheic key,value pairs of (Questions, Awnsers) 
#initalize tokenizer and use t5 as query generation model used to generate queries associated with input passages 
#Note: we are using a t5 model fine tuned for query generation as part of the BeIR Project 
logger.info('creating tokenizer and model to use in bi-encoder')
logger.info('creating model to use in bi-encoder')
#tokenizer  = T5Tokenizer.from_pretrained(GENQ_MODEL_NAME, legacy=False) 
qgen_model = T5ForConditionalGeneration.from_pretrained(GENQ_MODEL_NAME)
tokenizer = T5TokenizerFast.from_pretrained(GENQ_MODEL_NAME, do_lower_case=False)


#call eval() to force / ensure model is running in 'INFERENCE MODE' and not 'TRAINING' mode
logger.info('forcing model into eval mode')
qgen_model.eval()
model = qgen_model.to(device)
#tokenizer = tokenizer
print(device)

#initalize class to generate queries from passages
logger.info('initalize embedding querier')
queryer = query_ops(
     tokenizer
    , qgen_model 
    , SAVE_DIR
    , n_queries_per_passage = N_QUERIES_PER_PASSAGE
    , save_batch_size = 1000
    , train_batch_size = TOK_BATCH_SIZE    
    , return_tensors = return_tensors
    , padding =  padding
    , return_overflowing_tokens= return_overflow_tokens
    , max_seq_len = max_seq_len
    , truncation = truncation 
    , stride = stride 
    )

#generate query,passage key value pairs , save to disk , return paths 
logger.info('generating query, passage key value pairs')
query_passage_outpaths = queryer.gen_queries_from_passages(passages)

#create sentence_transformers comptable training dataset using InputExample() method from transformers
logger.info('creating training data for bi-encoder fine tuning')
pairs = queryer.create_training_data( query_passage_outpaths)

#create object to handle loading of InputExample() instances in batches of 50 
logger.info('creating loader to handle loading batches of data for model training')

#build and train the bi-encoder to be used for asymetric search (information retrieval)
#the trained model will encode passages into embeddings that are trained to be queried via short questions (as oppposed to just blindly taking the cossime between a short a long seq of text)
logger.info('building model')
ir_model = build_model(pairs
                    , BI_ENCODER_MODEL_NAME
                    , str(MODEL_SAVE_DIR / 'fine_tuned_biencoder')
                    , epochs=EPOCHS
                    , batch_size = BI_ENCODER_BATCH_SIZE
                    )



#build serachable index from all trained docs 
input_ids = queryer._tokenizer('where is egypt?',return_tensors='pt').input_ids
outputs = model.generate(input_ids)
print( queryer._tokenizer.decode(outputs[0], skip_special_tokens=True)
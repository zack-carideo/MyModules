import sys,os,logging, gc
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, T5Tokenizer,T5TokenizerFast, T5ForConditionalGeneration
from sentence_transformers import util , CrossEncoder
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
from util.misc import LoadCFG, seed_all, create_output_dirs
from util.data import load_data
from util.embedding_ops import query_ops
from util.model_ops import build_model , load_model
from util.index_ops import faiss_index 

#set seed 
SEED = 42
seed_all(SEED)

#load cfg params
cfg = LoadCFG(cfg_path, base_dir = root_).load()
DATA_PATH = cfg.data.input.data_path
SAVE_DIR = Path(cfg.data.output.data_save_dir)
MODEL_SAVE_DIR = Path(cfg.model.model_save_dir)
INDEX_SAVE_DIR = Path(cfg.model.ir.faiss_index.out_dir)

NSAMPS = cfg.model.n_samps
TOK_BATCH_SIZE = cfg.model.tokenizer.batch_size
BI_ENCODER_MODEL_NAME = cfg.model.ir.bi_encoder.model_name
EPOCHS = cfg.model.n_epochs
BI_ENCODER_BATCH_SIZE =  cfg.model.ir.bi_encoder.batch_size
CROSS_ENCODER_MODEL_NAME = cfg.model.ir.cross_encoder.model_name

#tokenizer setup
RETURN_TENSORS = cfg.model.tokenizer.return_tensors
PADDING =  cfg.model.tokenizer.padding
RETURN_OVERFLOW_TOKENS= cfg.model.tokenizer.return_overflow_tokens
MAX_SEQ_LEN = cfg.model.tokenizer.max_seq_len
TRUNCATION = cfg.model.tokenizer.truncation 
STRIDE = cfg.model.tokenizer.stride 

#query generator setup
GENQ_MODEL_NAME = cfg.model.query_gen.model_name 
N_QUERIES_PER_PASSAGE =  cfg.model.query_gen.n_queries_per_passage 

#device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#clean up gpu 
torch.cuda.empty_cache()
gc.collect()
logger.info(torch.cuda.memory_summary(device=DEVICE, abbreviated=True))

#create output dirs
create_output_dirs(SAVE_DIR, MODEL_SAVE_DIR)

    
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
passages = list(set(df['context']))[2300:2300+ NSAMPS]
passages = [(idx,txt) for idx,txt in enumerate(passages)]
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
model = qgen_model.to(DEVICE)
print(DEVICE)

#initalize class to generate queries from passages
logger.info('initalize embedding querier')
queryer = query_ops(
     tokenizer
    , qgen_model 
    , SAVE_DIR
    , n_queries_per_passage = N_QUERIES_PER_PASSAGE
    , save_batch_size = 1000
    , train_batch_size = TOK_BATCH_SIZE    
    , return_tensors = RETURN_TENSORS
    , padding =  PADDING
    , return_overflowing_tokens= RETURN_OVERFLOW_TOKENS
    , max_seq_len = MAX_SEQ_LEN
    , truncation = TRUNCATION 
    , stride = STRIDE 
    )

#generate query,passage key value pairs , save to disk , return paths 
logger.info('generating query, passage key value pairs')
query_passage_outpaths = queryer.gen_queries_from_passages(passages)
                  
#create sentence_transformers comptable training dataset using InputExample() method from transformers
logger.info('creating training data for bi-encoder fine tuning')

#create train df, including docidx, and chunk idx information
train_df , pairs= queryer.create_training_data( query_passage_outpaths)

#create object to handle loading of InputExample() instances in batches of 50 
logger.info('creating loader to handle loading batches of data for model training')

#build and train the bi-encoder to be used for asymetric search (information retrieval)
#the trained model will encode passages into embeddings that are trained to be queried via short questions (as oppposed to just blindly taking the cossime between a short a long seq of text)
logger.info('building model')
ir_model  = build_model(pairs
                    , BI_ENCODER_MODEL_NAME
                    , str(MODEL_SAVE_DIR / 'fine_tuned_biencoder')
                    , epochs=EPOCHS
                    , batch_size = BI_ENCODER_BATCH_SIZE
                    )

del ir_model
#build index to encode a fast query trained asyemetric embeddings
ir_model = load_model(str(MODEL_SAVE_DIR / 'fine_tuned_biencoder'), DEVICE)
ir_model.eval()


#create passage embeddings using the  new fine tuned bi - encoder
#define index object parameters
f_idx = faiss_index(train_df[['_index','passage']].drop_duplicates().reset_index(drop=True) #df
                    , ir_model #model
                    , ir_model[1].word_embedding_dimension
                    , text_col = 'passage'
                    , id_col = '_index'
                    , index_outpath = INDEX_SAVE_DIR
                    , cross_encoder_model_name = CROSS_ENCODER_MODEL_NAME
                    )
#create index
index_outp , data_outp, id_outp = f_idx.create_index()


#search index 
query_ = 'where is Ann Arbor?'
results = f_idx.search(query_,4, refine_with_crossencoder=True)


#del index and load to verify we can load static index and continue to query 
del f_idx

index_ ,data,_ids = faiss_index.load_index(index_outp, data_outp, id_outp)
_ce = CrossEncoder(CROSS_ENCODER_MODEL_NAME)
faiss_index.static_search(index_
                      , ir_model
                      , query_
                      , 5 
                      , data
                      , '_index'
                      , 'passage'
                      , _ce = _ce)



###
### CHEAP ALTERNATE INBUILT UTIILS TO EXZECUTE COSSIM
###
##query
# query_ = 'who is lady chapel?'
# query_emb = ir_model.encode(query_, convert_to_tensor=True)

# # We use cosine-similarity and torch.topk to find the highest 5 scores
# cos_scores = util.cos_sim(query_emb.to('cpu'), ir_doc_embeds)[0]
# top_results = torch.topk(cos_scores, k=6)

# print("\n\n======================\n\n")
# print("Query:", query_)
# print("\nTop 5 most similar sentences in corpus:")

# for score, idx in zip(top_results[0], top_results[1]):
#     print(pairs[idx], "(Score: {:.4f})".format(score))





# outputs = ir_model.encode([torch.Tensor(input_ids).type(torch.int64)])
# torch.Tensor(input_ids.input_ids).type(torch.int64).to('cuda')
# print( queryer._tokenizer.decode(outputs[0], skip_special_tokens=True))
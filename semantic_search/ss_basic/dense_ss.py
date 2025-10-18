#https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/ch08.html#overview_of_semantic_search_and_retriev

import torch 
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader ,TensorDataset
from datetime import datetime
from tqdm import tqdm 
from pathlib import Path 

def create_directory(directory_path):
    """
    Checks if a directory exists, and if it does not, creates one.
    
    Args:
        directory_path (str or Path): The path to the directory.
    """
    directory_path = Path(directory_path)
    if not directory_path.exists():
        directory_path.mkdir(parents=True, exist_ok=True)


#create custom dataset class
class MyDataset(Dataset):
    def __init__(self,tensor_data):
        self.data = tensor_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def batch_and_process(list, batch_size):

    """
    This function takes a list and a batch size and returns a generator that will yield batches of the list with the specified size.
    Args:
        list (list): The list to batch.
        batch_size (int): The size of the batches.
    Returns:
        generator: A generator that yields batches of the list.
    """
    
    for idx in range(0, len(list), batch_size):
        yield list[idx:idx + batch_size]



def generate_runid():
    """
    Generates a run ID based on the current date in the format mm-dd-yy.
    
    Returns:
        str: The generated run ID.
    # Example usage:
        run_id = generate_runid()
        print(f"Run ID: {run_id}")

    """
    return datetime.now().strftime("%m-%d-%y")


def list_files(directory_path):
    """
    Lists all files in the given directory.
    
    Args:
        directory_path (str or Path): The path to the directory.
        
    Returns:
        list: A list of file paths.
    """
    directory_path = Path(directory_path)
    return [file for file in directory_path.iterdir() if file.is_file()]


#method2 memory mapped tensor 
batch_size = 500

tensor_outpath = Path("/home/zjc1002/Mounts/data/trash/corpus_embeddings.bin")
parent_outdir = tensor_outpath.parent
model_path = '/home/zjc1002/Mounts/llms/sentence-transformers_all-mpnet-base-v2'
convert_to_tensor = True 
model_kwargs = None #{"torch_dtype": "float16"}


corpus = [
    "A man is eating food.",
    "A man is eating a piece of bread.",
    "The girl is carrying a baby.",
    "A man is riding a horse.",
    "A woman is playing violin.",
    "Two men pushed carts through the woods.",
    "A man is riding a white horse on an enclosed ground.",
    "A monkey is playing drums.",
    "A cheetah is running behind its prey.",
] * 400

# Query sentences:
queries = [
    "A man is eating pasta.",
    "Someone in a gorilla costume is playing a set of drums.",
    "A cheetah chases prey on across a field.",
]

# load model 
model = SentenceTransformer(model_path, model_kwargs=model_kwargs) 
max_seq_length = model.max_seq_length 

_run_id = generate_runid()
run_outdir = parent_outdir / _run_id
for idx, _batch in enumerate(batch_and_process(corpus, batch_size)):
    _chunk_fpath = run_outdir / f"embeddings_{idx}.bin"
    _embeds = model.encode(_batch, convert_to_tensor=convert_to_tensor)
    print(idx)
    create_directory(run_outdir )
    torch.save(_embeds, _chunk_fpath.as_posix())
    print(len(_batch))



#load the chunks and process 
query_embeds = model.encode(queries, convert_to_tensor=convert_to_tensor)
sim_scores = []
for file in run_outdir.iterdir():
     if file.is_file(): 
        _embeds = torch.load(file)

        sim_scores.append(util.semantic_search(query_embeds
                                            , _embeds
                                            , score_function=util.dot_score)
                                            )
                                            
        print(file)

















# Use "convert_to_tensor=True" to keep the tensors on GPU (if available)
corpus_embeddings = model.encode(corpus*100, convert_to_tensor=convert_to_tensor)
query_embeddings = model.encode(queries, convert_to_tensor=convert_to_tensor)
query_embeddings= util.normalize_embeddings(query_embeddings)
corpus_embeddings= util.normalize_embeddings(corpus_embeddings)


#
#save embeddings then  we load them and process in batches
#

#create dataset
dataset = MyDataset(corpus_embeddings)

#create dataloader 
dataloader = torch.utils.data.DataLoader(dataset, batch_size=500, shuffle=True)

for batch in dataloader:
    print(batch.shape)

#save the embeddings to disk
#tensor_outpath = Path("/home/zjc1002/Mounts/data/trash/corpus_embeddings.bin")
#torch.save(corpus_embeddings, tensor_outpath)





#method2 memory mapped tensor 
tensor_outpath = Path("/home/zjc1002/Mounts/data/trash/corpus_embeddings.bin")
tensor_outdir = tensor_outpath.parent
sims_outpath = tensor_outdir / "sims.bin"

rows, cols = corpus_embeddings.shape[0], corpus_embeddings.shape[1]

#create the file with shared=True to enable cross-process memory sharing
samples = torch.FloatTensor(
    torch.FloatStorage.from_file(tensor_outpath.as_posix(), shared=True, size = rows*cols)
    ).reshape( rows, cols)  

#populate the samples data (every in-place assignment to samples will be reflected in the file)
for idx in tqdm(range(rows)):
    samples[idx] = corpus_embeddings[idx]


#
#now load it 
#

tensor_inpath = tensor_outpath

# shared=False prevents changes to samples from affecting the data on disk

samples = torch.FloatTensor(
    torch.FloatStorage.from_file(tensor_inpath
                                 , shared=False
                                 , size= rows * cols)
                                 ).reshape( rows, cols)

dataset = TensorDataset(samples)
loader = DataLoader(dataset, batch_size=2, num_workers=0)

sim_scores = []
for idx, batch in tqdm(enumerate(loader)):
    
    # batch is a (256, 32, 30) tensor
    print(batch[0].shape)
    sim_scores.append(util.semantic_search(query_embeddings
                                           , batch[0]
                                           , score_function=util.dot_score)
                                           )
    

    pass




#### 
#STOP
####


##########
##INPUTS##
##########

# Corpus with example sentences
corpus = [
    "A man is eating food.",
    "A man is eating a piece of bread.",
    "The girl is carrying a baby.",
    "A man is riding a horse.",
    "A woman is playing violin.",
    "Two men pushed carts through the woods.",
    "A man is riding a white horse on an enclosed ground.",
    "A monkey is playing drums.",
    "A cheetah is running behind its prey.",
]

# Query sentences:
queries = [
    "A man is eating pasta.",
    "Someone in a gorilla costume is playing a set of drums.",
    "A cheetah chases prey on across a field.",
]

model_path = '/home/zjc1002/Mounts/llms/sentence-transformers_all-mpnet-base-v2'
convert_to_tensor = True 
model_kwargs = None #{"torch_dtype": "float16"}


##########
###CODE###
##########
#load the model
model = SentenceTransformer(model_path, model_kwargs=model_kwargs) 
max_seq_length = model.max_seq_length 

# Use "convert_to_tensor=True" to keep the tensors on GPU (if available)
corpus_embeddings = model.encode(corpus, convert_to_tensor=convert_to_tensor)
query_embeddings = model.encode(queries, convert_to_tensor=convert_to_tensor)


corpus_embeddings = corpus_embeddings.to("cpu")
corpus_embeddings = util.normalize_embeddings(corpus_embeddings)

query_embeddings = query_embeddings.to("cpu")
query_embeddings = util.normalize_embeddings(query_embeddings)
hits = util.semantic_search(query_embeddings, corpus_embeddings, score_function=util.dot_score)









import torch 
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModel

class SemanticSearch:

    def __init__(self
                 , model_path
                 , index_outpath
                 , convert_to_tensor=True
                 , model_kwargs=None
                 , device='cpu'
                 
                 ):
        
        self.model_path = model_path
        self.index_outpath = index_outpath 
        self.convert_to_tensor = convert_to_tensor
        self.model_kwargs = model_kwargs if model_kwargs else {}
        self.device=device

        self.model = None
        self.max_seq_length  = None
        self.corpus_embeds = None
        self.query_embeds = None

    def load_model(self):
        self.model = SentenceTransformer(self.model_path, model_kwargs = self.model_kwargs)
        self.max_seq_length = self.model.max_seq_length

    # def load_index(self):
    #     self.corpus_embeds = torch.load(self.index_outpath)
    
    # def save_index(self):
    #     print(f'loading index from {self.index_outpath}')
    #     torch.save(self.corpus_embeds, self.index_outpath)

    def generate_embeddings(self,corpus):
        if not self.model:
            raise ValueError("Model not loaded. Call load_model() first.")
        return self.model.encode(corpus, convert_to_tensor=self.convert_to_tensor)

    def create_index(self,corpus):
        if not self.model:
            raise ValueError("Model not loaded. Call load_model() first.")
        self.corpus_embeds = self.generate_embeddings(corpus)    




    def query_embeddings(self,corpus, queries):
        
        if not self.model:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        #generate embeddings 
        self.query_embeds = self.model.encode(queries, convert_to_tensor=self.convert_to_tensor)
        self.corpus_embeds = self.model.encode(corpus, convert_to_tensor= self.convert_to_tensor)

        #send corpus embeddings to device and normalize for unit length 
        self.corpus_embeds = self.corpus_embeds.to(self.device)
        self.corpus_embeds = util.normalize_embeddings(self.corpus_embeds)
        
        #send query  embeddings to device and normalize for unit length 
        self.query_embeds = self.query_embeds.to(self.device)
        self.query_embeds = util.normalize_embeddings(self.query_embeds)
        
        #execute semantic search 

        hits = util.semantic_search(self.query_embeds
                                    , self.corpus_embeds
                                    , score_function=util.dot_score)
        
        return hits

# Example usage:
model_path = '/home/zjc1002/Mounts/llms/sentence-transformers_all-mpnet-base-v2'
corpus = [
    "A man is eating food.",
    "A man is eating a piece of bread.",
    "The girl is carrying a baby.",
    "A man is riding a horse.",
    "A woman is playing violin.",
    "Two men pushed carts through the woods.",
    "A man is riding a white horse on an enclosed ground.",
    "A monkey is playing drums.",
    "A cheetah is running behind its prey.",
]
queries = [
    "A man is eating pasta.",
    "Someone in a gorilla costume is playing a set of drums.",
    "A cheetah chases prey on across a field.",
]
convert_to_tensor = True
model_kwargs = None #{"torch_dtype": "float16"}

semantic_search = SemanticSearch(model_path,  convert_to_tensor, model_kwargs)
semantic_search.load_model()
hits = semantic_search.query_embeddings(corpus, queries)
print(hits)










####
#### THESE ARE THE BEST FUNCTIONS TO USE ####
####


def load_model(model_path, model_kwargs):

    model = SentenceTransformer(model_path, model_kwargs = model_kwargs)
    max_seq_length = model.max_seq_length
    return  model, max_seq_length 


def generate_embeddings(model,corpus,convert_to_tensor, normalize = True):

    if not model:
        raise ValueError("Model not loaded. Call load_model() first.")
    
    _embeddings = model.encode(corpus, convert_to_tensor=convert_to_tensor)

    if normalize: 
        _embeddings = util.normalize_embeddings(_embeddings)

    return _embeddings


def query_embeddings(model, corpus, queries,convert_to_tensor=True,normalize=True):

    if not model:
        raise ValueError("Model not loaded. Call load_model() first.")
    
    query_embeds = generate_embeddings(model, queries, convert_to_tensor=convert_to_tensor,normalize=normalize)
    corpus_embeds = generate_embeddings(model, corpus, convert_to_tensor=convert_to_tensor,normalize=normalize)

        

    #execute semantic search 

    hits = util.semantic_search(query_embeds
                                , corpus_embeds
                                , score_function=util.dot_score)
    
    return hits
####
#### THESE ARE THE BEST FUNCTIONS TO USE ####
#### END

# Example usage:
model_path = '/home/zjc1002/Mounts/llms/sentence-transformers_all-mpnet-base-v2'
corpus = [
    "A man is eating food.",
    "A man is eating a piece of bread.",
    "The girl is carrying a baby.",
    "A man is riding a horse.",
    "A woman is playing violin.",
    "Two men pushed carts through the woods.",
    "A man is riding a white horse on an enclosed ground.",
    "A monkey is playing drums.",
    "A cheetah is running behind its prey.",
]
queries = [
    "A man is eating pasta.",
    "Someone in a gorilla costume is playing a set of drums.",
    "A cheetah chases prey on across a field.",
]
convert_to_tensor = True
model_kwargs = None #{"torch_dtype": "float16"}


model, max_seq_len = load_model(model_path, model_kwargs)



query_embeddings(model, corpus, queries,convert_to_tensor=True,normalize=True)


semantic_search = SemanticSearch(model_path,  convert_to_tensor, model_kwargs)
semantic_search.load_model()
semantic_search.generate_embeddings(corpus)
hits = semantic_search.query_embeddings(queries)
print(hits)
















#model = SentenceTransformer('/home/zjc1002/Mounts/llms/sentence-transformers_all-mpnet-base-v2')
#text1 = 'i am zack'
#text2 = 'you are sam'
#convert_to_tensor = True
# Get the embeddings
#embeddings1 = model.encode(text1, convert_to_tensor=convert_to_tensor)
#embeddings2 = model.encode(text2, convert_to_tensor=convert_to_tensor)


def calc_sim(model, text1, text2, convert_to_tensor = True):

    # Get the embeddings
    embeddings1 = model.encode(text1, convert_to_tensor=convert_to_tensor)
    embeddings2 = model.encode(text2, convert_to_tensor=convert_to_tensor)
    
    # Compute the cosine similarity
    sim = torch.nn.functional.cosine_similarity(embeddings1, embeddings2, dim=0)
    
    return sim.item()



def tensor_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Computes the cosine similarity between two tensors.
    Args:
        a (torch.Tensor): The first tensor.
        b (torch.Tensor): The second tensor.
    Returns:
        torch.Tensor: The cosine similarity between the two input tensors.
    """
    
    return  torch.dot(a, b) / (torch.norm(a) * torch.norm(b))


def tensor_batch_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Computes the cosine similarity between two batches of tensors.
    Args:
        a (torch.Tensor): The first batch of tensors.
        b (torch.Tensor): The second batch of tensors.
    Returns:
        torch.Tensor: The cosine similarity between the two batches of tensors.
    """
    
    return torch.mm(a, b.T) / (torch.norm(a, dim=1)[:, None] * torch.norm(b, dim=1)[None, :])


def tensor_topk_sim(a: torch.Tensor, b: torch.Tensor, k: int) -> torch.Tensor:
    """
    Computes the top-k cosine similarity between two tensors.
    Args:
        a (torch.Tensor): The first tensor.
        b (torch.Tensor): The second tensor.
        k (int): The number of top-k elements to return.
    Returns:
        torch.Tensor: The top-k cosine similarity between the two input tensors.
    """
    
    sim = torch.mm(a, b.T) / (torch.norm(a, dim=1)[:, None] * torch.norm(b, dim=1)[None, :])
    
    return torch.topk(sim, k, dim=1)










from sentence_transformers import SentenceTransformer
from datetime import datetime

# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")
max_seq_length = model.max_seq_length 

# The sentences to encode
sentences = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium.",
]

# 2. Calculate embeddings by calling model.encode()
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# 3. Calculate the embedding similarities
similarities = model.similarity(embeddings, embeddings)
print(similarities)
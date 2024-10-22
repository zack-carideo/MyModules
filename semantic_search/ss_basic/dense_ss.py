#https://learning.oreilly.com/library/view/hands-on-large-language/9781098150952/ch08.html#overview_of_semantic_search_and_retriev

import torch 
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModel

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
model_kwargs = {"torch_dtype": "float16"}


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









class SemanticSearch:

    def __init__(self
                 , model_path
                 , convert_to_tensor=True
                 , model_kwargs=None
                 , device='cpu'):
        
        self.model_path = model_path
        self.convert_to_tensor = convert_to_tensor
        self.model_kwargs = model_kwargs if model_kwargs else {}
        self.device=device

        self.model = None
        self.max_seq_length  = None
        self.corpus_embeddings = None
        self.query_embeds = None

    def load_model(self):
        self.model = SentenceTransformer(self.model_path, model_kwargs = self.model_kwargs)
        self.max_seq_length = self.model.max_seq_length

    def generate_embeddings(self,corpus):
        if not self.model:
            raise ValueError("Model not loaded. Call load_model() first.")
        return self.model.encode(corpus, convert_to_tensor=self.convert_to_tensor)

    def query_embeddings(self,queries):
        
        if self.corpus_embeddings is None:
            raise ValueError("Embeddings not generated. Call generate_embeddings() first.")
        if not self.model:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        self.query_embeds = self.model.encode(queries, convert_to_tensor=self.convert_to_tensor)

        #send corpus embeddings to device and normalize for unit length 
        self.corpus_embeddings = self.corpus_embeddings.to(self.device)
        self.corpus_embeddings = util.normalize_embeddings(self.corpus_embeddings)
        
        #send query  embeddings to device and normalize for unit length 
        self.query_embeds = self.query_embeds.to(self.device)
        self.query_embeds = util.normalize_embeddings(self.query_embeds)
        
        #execute semantic search 

        hits = util.semantic_search(self.query_embeds
                                    , self.corpus_embeddings
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
semantic_search.generate_embeddings(corpus)
hits = semantic_search.query_embeddings(queries)
print(hits)















def load_model(model_path, model_kwargs):

    model = SentenceTransformer(model_path, model_kwargs = model_kwargs)
    max_seq_length = model.max_seq_length
    return  model, max_seq_length 


def generate_embeddings(model,corpus,convert_to_tensor):

    if not model:
        raise ValueError("Model not loaded. Call load_model() first.")
    
    _embeddings = model.encode(corpus, convert_to_tensor=convert_to_tensor)
    return _embeddings


def query_embeddings(model, corpus, queries,convert_to_tensor=True,normalize=True):

    if not model:
        raise ValueError("Model not loaded. Call load_model() first.")
    

    query_embeds = generate_embeddings(model, queries, convert_to_tensor=convert_to_tensor)
    corpus_embeds = generate_embeddings(model, corpus, convert_to_tensor=convert_to_tensor)

    if normalize: 
        query_embeds = util.normalize_embeddings(query_embeds)
        corpus_embeds = util.normalize_embeddings(corpus_embeds)
        

    #execute semantic search 

    hits = util.semantic_search(query_embeds
                                , corpus_embeds
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
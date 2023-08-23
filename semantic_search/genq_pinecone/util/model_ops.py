import sentence_transformers
from sentence_transformers import models, SentenceTransformer, losses,InputExample,datasets
from typing import List, Tuple 

def gen_model_data_loader( pairs: List[Tuple], batch_size = 5):
    """_summary_

    Args:
        pairs (_type_): list of (query,passage) pairs to be used in fine tuning of a bi-encoder 
        batch_size (int, optional): The `batch_size` parameter determines the number of query-passage pairs that will be processed
                                    together in each iteration during the generation
                                    of queries from passages. It controls how many pairs are processed in parallel, which can help
                                    improve efficiency and speed up the process.
                    

    Returns:
        _type_: _description_
    """
    
    # `InputExample` is a class provided by the `sentence_transformers`
    # library. It represents a single training example for a sentence
    # transformer model. It consists of a list of texts, where each text
    # represents a query or a passage. In the context of the `query_ops`
    # class, `InputExample` is used to create pairs of queries and passages
    # for training a sentence transformer model.
    #InputExample(texts=[q, p])
    pairs_ = [InputExample(texts=[t[0], t[1]]) for t in pairs]
    
    #load the pairs into a NoDuplicatesDataLoader. 
    # We use the no duplicates data loader to avoid placing duplicate passages in the same batch
    # as this will confuse the ranking mechanism of MNR loss.
    loader = datasets.NoDuplicatesDataLoader(pairs_, batch_size = batch_size)
    
    return loader     
    
def create_bi_encoder(bi_encoder_model_name:str ):

    """
    We are initializing from a pretrained MPNet model, which by default outputs 512 embeddings.
    The second module is a mean pooling layer that takes the average activations across all of 
    these embeddings to create a single sentence embedding.
    
    SentenceTransformer(modules=[mpnet, pooler])` is creating an instance of the
    `SentenceTransformer` class. It takes two modules, `mpnet` and `pooler`, as arguments. These
    modules are used to transform input sentences into fixed-length sentence embeddings. The `mpnet`
    module is a pretrained transformer model that outputs word embeddings, and the `pooler` module
    is a mean pooling layer that takes the average activations across all word embeddings to create
    a single sentence embedding. The `SentenceTransformer` class combines these modules to create a
    bi-encoder model that can encode pairs of sentences.
    
    Args: 
        bi_encoder_model_name: name of the pretrained huggyface/sentence bi-encoder model to use as baseline for fine tuning
    
    Returns:
        _type_: _description_
    """
    #baseline bi-encoder
    mpnet = models.Transformer(bi_encoder_model_name)
    
    # define how we want to generate sentence embeddings from input sequence embeddings
    # for the modules parameter, we pass a list of layers which are executed consecutively. 
    # Input text are first passed to the first entry (word_embedding_model). 
    # The output is then passed to the second entry (pooling_model), which then returns our sentence embedding.
    
    pooler = models.Pooling(
        mpnet.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True)

    return SentenceTransformer(modules=[mpnet, pooler])



def create_loss_function(model: sentence_transformers.SentenceTransformer):
    """
    `losses.MultipleNegativesRankingLoss` is a loss function used for training the
    bi-encoder model. It is specifically designed for training models that generate sentence
    embeddings.

    Args:
        model (sentence_transformers.SentenceTransformer): sentence transformer bi-encoder model to fine tune 

    Returns:
        _type_: _description_
    """

    return losses.MultipleNegativesRankingLoss(model)


def build_model(query_passage_pairs
                ,bi_encoder_model_name, model_outpath ,epochs=3, batch_size = 2):
    
    #create data loader 
    loader = gen_model_data_loader( query_passage_pairs, batch_size = batch_size)
    
    #generate model grid/frame
    model = create_bi_encoder(bi_encoder_model_name)
    
    #create loss function to evalute model with 
    loss = create_loss_function(model)
    
    warmup_steps = int(len(loader)*epochs*.1)
    
    model.fit(
        train_objectives = [(loader,loss)]
       , epochs = epochs
       , warmup_steps = warmup_steps
       , output_path = model_outpath
       , show_progress_bar = True


    )

    return model 
    
    
    
    
    

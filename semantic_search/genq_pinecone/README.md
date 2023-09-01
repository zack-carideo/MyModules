#### E2E Demo Augmented Query Geneation and Asymetric  Search with Cross Encoder refinement

    0) Tokenization of long passages using window and stride
        - 
    1) Unsupervised Query Generation 
        - 'BeIR/query-gen-msmarco-t5-large-v1'

    2) Fine Tuning Bi-Encoder to use for semantic search 
        - 'msmarco-distilroberta-base-v3'

    3) Creation of Faiss index using bi-encoder encoded passages
    4) Incorporation of Cross-Encoder on top of results returned from Bi-Encoder retreival
        - 'cross-encoder/ms-marco-TinyBERT-L-2-v2' 



### RESOURCES 
- THIS MODULE IS A Variant of the IMPLEMENTATION PROVIDED HERE: https://www.pinecone.io/learn/series/nlp/genq/
- https://huggingface.co/docs/transformers/model_doc/t5




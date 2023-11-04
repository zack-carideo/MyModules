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


### Run Instructions 
- **Option 1(run from terminal)**: python intro.py
- **Option 2(Interactive)**: demo.ipynb 
    -    demo.py walks through  process used to link original text chunks to post tokenized spanning chunks and provides examples how data after each step of the framework 



### RESOURCES 
- THIS MODULE IS A Variant of the IMPLEMENTATION PROVIDED HERE: https://www.pinecone.io/learn/series/nlp/genq/
- https://huggingface.co/docs/transformers/model_doc/t5




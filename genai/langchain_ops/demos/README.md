### This repository (langchain_ops/demos) contains a series of interactive demos to illustrate a variety of LLM-based operations to execute RAG-based components independently and in an end-to-end fashion 

- End-to-End Rag Demo 

    - **rag/Rag_with_MapReduce.ipynb:**
        - **Objective:** illustrate how multiple LLM models can be integrated and used sequentially to efficiently execute each component of the RAG framework within langchain using a combination of small retrievers, and highly compressed llamaccp models for geneation. 

        - Core Components 
            - **Merged Retriever:** An ensemble of small retrievers is used to index the source document text chunks. each index represents a different models embedding representation of the text. These vector indcies are quried by end user to identify the top n relevant documents associated with an end user's query 
                - Langchain's MergedRetriever is used to facilitate the ensembling of multiple retrievers 
                - GPT-2 is used to identify potential semantic duplicates returned from the merged retriever and filter them prior to any generation. 

            - **MapReduce & Generation with LlamaCpp:** The use of a foundational LLM for generation makes a HUGE difference. However, many of these models cannot be used on a CPU or household GPU. Llamacpp is a framework that leverages gguf model files that drastically reduce the footprint required to conduct inference with llama2 models. 
                - LlamaCpp is used to facilitate the map reduce process and generate the final contextual summarization of retriever's top n candidate responses. 

         End-to-end example of using an ensemble of small/efficient retrievers to identify the most relevant content from an end user's query. The top n documents returned from the retriever framework are passed to a map reduce pipeline to then

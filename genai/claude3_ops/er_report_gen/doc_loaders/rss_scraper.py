import transformers, feedparser, logging 
from langchain_community.document_loaders import RSSFeedLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
import pandas as pd

#set up basic logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
logger =  logging.getLogger(__name__)


def parse_rss_feeds(feed_urls: list 
    , chunk_docs: bool = False
    , llm_dir: str = "/home/zjc1002/Mounts/llms/"
    , model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    , device: str = 'cuda'
    , out_path: str = None 
    ):
    

    """Parses RSS feeds from the given URLs and optionally applies semantic chunking to the documents.
        Args:
            feed_urls (list): List of RSS feed URLs to parse.
            chunk_docs (bool, optional): Whether to apply semantic chunking to the documents. Defaults to False.
            llm_dir (str, optional): Directory where the language model is stored. Defaults to "/home/zjc1002/Mounts/llms/".
            model_name (str, optional): Name of the model to use for embeddings. Defaults to "sentence-transformers/all-MiniLM-L6-v2".
            device (str, optional): Device to run the model on ('cuda' or 'cpu'). Defaults to 'cuda'.
            out_path (str, optional): Path to save the output DataFrame as a parquet file. Defaults to "/home/zjc1002/Mounts/data/emerging_risks/rss_data.pq".
        
    
    """
    
    #Define RSS Feed Loader
    loader = RSSFeedLoader(urls=feed_urls)

    #do you want to apply semantic chunking to the documents?
    if chunk_docs: 

        #load embeddings to use for semantic chunking
        embeddings_ =  HuggingFaceEmbeddings(model_name=model_name
                                            , model_kwargs = {'device':device}
                                            , cache_folder=llm_dir
                                            )

        # This chunker works by determining when to "break" apart sentences. 
        # This is done by looking for differences in embeddings between any two sentences. 
        # When that difference is past some threshold, then they are split.
        text_splitter = SemanticChunker(embeddings=embeddings_
                                        , breakpoint_threshold_type = 'gradient'
                                        )

        #chunk docs using semantic simlarity between sentences 
        docs  = loader.load_and_split(text_splitter=text_splitter)

    else:
        
        #extract data from rss feeds
        docs = loader.load()


    #extract 2 elements from each document and create a list of dictionaries to create a dataframe
    _meta = [docs[idx].metadata for idx in range(len(docs))]
    _content= [docs[idx].page_content for idx in range(len(docs))]

    d_list = []
    for idx,_d in enumerate(_meta):
        _d['content'] = _content[idx]
        d_list.append(_d)

    #generate dataframe of news articles and regualtory documents
    news_df = pd.DataFrame(d_list).drop_duplicates(
        subset=['content'])
    
    if out_path:
        assert out_path.endswith('.pq'), "Output path must be a parquet file"
        news_df.to_parquet(out_path)
        logger.info(f"rss data saved to {out_path}")
        return out_path
    else: 
        return news_df
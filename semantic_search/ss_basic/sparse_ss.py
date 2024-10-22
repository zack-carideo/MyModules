
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction import _stop_words
import string
import tqdm 
from typing import List, Set
import numpy as np 

def bm25_tokenizer(text: str
                   , stop_words: Set[str] = None
                   , punctuations: str = None
                   ) -> List[str]:
    
    """
    Tokenizer for BM25.
    
    This function tokenizes the input text by converting it to lowercase, removing punctuation, and filtering out stop words. 
    It returns a list of tokens that can be used for BM25 ranking.

    Parameters:
        text (str): The input text to be tokenized.
            Example: "The quick brown fox jumps over the lazy dog."
        stop_words (Set[str], optional): A set of stop words to be filtered out from the tokenized text. Defaults to _stop_words.ENGLISH_STOP_WORDS.
            Example: {"the", "over", "and"}
        punctuations (str, optional): A string of punctuation characters to be stripped from the tokens. Defaults to string.punctuation.
            Example: "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"

    Returns:
        List[str]: A list of tokens after processing the input text.
    
    Example: 
        bm25_tokenizer(["quick", "brown", "fox", "jumps", "lazy", "dog"]
                   , stop_words: Set[str] = _stop_words.ENGLISH_STOP_WORDS
                   , punctuations: str = string.punctuation
                   ) -> List[str]:                   
    """

    tokenized_doc = []
    for token in text.lower().split():
        token = token.strip(punctuations) if punctuations else token

        if stop_words: 
            if len(token) > 0 and token not in stop_words:
                tokenized_doc.append(token)
        else: 
            if len(token) > 0:
                tokenized_doc.append(token)

    return tokenized_doc

def tokenize_corpus(texts: List[str] 
                    , punctuations: str = None 
                    , stop_words: Set[str] = None
                    ) -> List[List[str]]:

    """
    Tokenizes a list of text passages.
    Args:
        texts (List[str]): A list of text passages to be tokenized.
        punctuations (str, optional): A string of punctuation characters to be removed from the text. Defaults to None.
        stop_words (Set[str], optional): A set of stop words to be removed from the text. Defaults to None.
    Returns:
        List[List[str]]: A list of tokenized text passages, where each passage is represented as a list of tokens.
    """
    
    tokenized_corpus = []
    for passage in tqdm.tqdm(texts):
        tokenized_corpus.append(bm25_tokenizer(passage
                                               , punctuations=punctuations
                                               , stop_words=stop_words))

    return tokenized_corpus

def build_sparse_search_index(tokenized_corpus: List[str]) -> BM25Okapi:
    
    bm25 = BM25Okapi(tokenized_corpus)

    return bm25

def keyword_search(query: str
                   , bm25_index: BM25Okapi
                   , top_k: int = 3
                   , num_candidates: int = 15
                   , punctuations: str = None
                   , stop_words: Set[str] = None
                   ) -> List[dict]:
    """
    Perform a keyword search using the BM25 index.

    This function takes a query string, tokenizes it, and searches the BM25 index to find the most relevant documents.
    It returns the top-k results based on the BM25 scores.

    Parameters:
        query (str): The input query string to search for.
            Example: "lazy dog"
        bm25_index (BM25Okapi): The BM25 index built from the corpus.
        top_k (int, optional): The number of top results to return. Defaults to 3.
        num_candidates (int, optional): The number of candidate results to consider before selecting the top-k. Defaults to 15.

    Returns:
        List[dict]: A list of dictionaries containing the corpus_id and score of the top-k results.
    
    Example:
        keyword_search("lazy dog", bm25_index, top_k=3, num_candidates=15) -> [{'corpus_id': 0, 'score': 1.5}, ...]
    """
    print("Input question:", query)

    # Encode query (using same tokenizer used to generate bm25 index being searched)
    encoded_query = bm25_tokenizer(query, punctuations=punctuations, stop_words=stop_words)

    # Search index on query 
    bm25_scores = bm25_index.get_scores(encoded_query)

    # Return top-n results 
    top_n = np.argpartition(bm25_scores, -num_candidates)[-num_candidates:]
    
    # Sort top-n results by score
    bm25_hits = [{'corpus_id': idx, 'score': bm25_scores[idx]} for idx in top_n]
    bm25_hits = sorted(bm25_hits, key=lambda x: x['score'], reverse=True)
    
    print(f"Top-3 lexical search (BM25) hits")
    for hit in bm25_hits[0:top_k]:
        print("\t{:.3f}\t{}".format(hit['score'], texts[hit['corpus_id']].replace("\n", " ")))

    return bm25_hits


if __name__ == "__main__":
    
    #only use preprocessing if strings are very long or if you have a lot of data
    punctuations = string.punctuation
    stop_words = _stop_words.ENGLISH_STOP_WORDS

    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "A quick brown dog outpaces a quick fox.",
        "The quick fox is quick.",
        "The dog is lazy."
    ]

    tokenized_corpus = tokenize_corpus(texts
                                       , punctuations=None
                                       , stop_words=None)
    
    bm25_index = build_sparse_search_index(tokenized_corpus)
    
    results = keyword_search("who is the quick fox", bm25_index, top_k=4, num_candidates=4)
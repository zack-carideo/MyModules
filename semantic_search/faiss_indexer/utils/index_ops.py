import faiss , torch 
import pandas as pd 
#
#USER QUERY INPUT SPECIFIC FUNCS 
#
def index_embeddings(sentence_embeddings: 'np.ndarray', index=None):
    #create vector index
    vector_dimension = sentence_embeddings.shape[1]
    index = faiss.IndexFlatL2(vector_dimension)
    faiss.normalize_L2(sentence_embeddings.numpy())
    index.add(sentence_embeddings)
    return index 

    
def query_index(processer, index, sentence_d, query_text: str):
    
    sv_embed = torch.vstack([se.to('cpu') for se in processer.e2e([query_text])])

    #normalize embedding for querty length 
    faiss.normalize_L2(sv_embed.numpy())
    
    #identify total elements in index
    k = index.ntotal
    
    #calculate distances of the query to each passage in index
    distances, ann = index.search(sv_embed, k=k)
    
    #save results to df 
    results = pd.DataFrame({'distances': distances[0], 'ann': ann[0]})
    df_ = pd.DataFrame.from_dict(sentence_d, orient='index').reset_index()
    df_['doc'] = df_.apply(lambda x: x['index'].split('_')[0], axis=1)
    df_['sent'] = df_.apply(lambda x: x['index'].split('_')[1], axis=1)
    df_.sort_values(by=['sent','doc']).reset_index(drop=True)
    results_df = pd.merge(
        results.reset_index(drop=True), 
        df_    , 
        left_on='ann', 
        right_index=True
        )
    results_df['query_txt']= query_text 
    return results_df 

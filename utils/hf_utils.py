"""
THIS FILE CONTAINS FUNCTIONS USED TO FACILITATE EASY DOWNLOADING / LOADING / SAVING / SERVING MODELS FROM HuggingFace 
"""
import os
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download 

def download_model_hfhub(repo_id
    , filename
    , repo_type
    , local_dir
    , local_dir_use_symlinks
    ):
    
    """
    download a model from huggingface hub and save to local directory

    Args:
        repo_id (str): repo_id of the model
        filename (str): filename of the model       
        repo_type (str): repo_type of the model
        local_dir (str): local_dir of the model
        local_dir_use_symlinks (bool): local_dir_use_symlinks of the model
        Returns:
            local_dir (str): local_dir of the model
    Ex: 
        repo_id = "shaowenchen/llama-2-7b-langchain-chat-gguf"
        filename = 'llama-2-7b-langchain-chat.Q4_K.gguf'
        repo_type = "model"
        local_dir = "/home/zjc1002/Mounts/llms/llama-2-7b-langchain-chat-gguf"
        local_dir_use_symlinks = False

        download_model_hfhub(repo_id, filename, repo_type, local_dir, local_dir_use_symlinks)
    """

    # Check if local_dir exists
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

        #download a model 
        hf_hub_download(
            repo_id = repo_id
            , filename = filename
            , repo_type = repo_type
            , local_dir = local_dir
            , local_dir_use_symlinks = local_dir_use_symlinks
            )
        return Path(local_dir, filename) 
    else:
        print(f"local_dir:{local_dir} already exists, skipping download")



def download_full_model_directory(repo_id, local_dir):
    """
    Download the full model directory from HuggingFace hub and save to local directory

    Args:
        repo_id (str): repo_id of the model
        local_dir (str): local_dir to save the model
    Returns:
        local_dir (str): local_dir where the model is saved
    Ex: 
        repo_id = "sentence-transformers/msmarco-MiniLM-L-6-v3"
        local_dir = "/home/zjc1002/Mounts/llms/msmarco-MiniLM-L-6-v3"

        download_full_model_directory(repo_id, local_dir)
    """
    
    # Check if local_dir exists
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    # Download the full model directory
    snapshot_download(repo_id=repo_id, local_dir=local_dir)
    return local_dir

if __name__ == '__main__':
    # Example usage
    repo_id = "sentence-transformers/msmarco-MiniLM-L-6-v3"
    local_dir = "/home/zjc1002/Mounts/llms/msmarco-MiniLM-L-6-v3"
    download_full_model_directory(repo_id, local_dir)


    
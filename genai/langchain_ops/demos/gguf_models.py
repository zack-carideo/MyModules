from ctransformers import AutoModelForCausalLM

import yaml

# Define the hardcoded string and numeric variables
variables = {
    'instructions':'create a new file called test.py and insert a python script to download the smallest version of llama2 chat from huggingface.com to the current directory'
    'llm_path': "/home/zjc1002/Mounts/llms/llama-2-13b-langchain-chat-gguf/llama-2-13b-langchain-chat.Q4_1.gguf",
    'model_file': "llama-2-13b-langchain-chat.Q4_1.gguf",
    'model_type': "llama",
    'gpu_layers': 10,
    'use_internet': 0
}

# Unpack the key-value pairs into individual variables
instructions = variables['instructions']
llm_path = variables['llm_path']
model_file = variables['model_file']
model_type = variables['model_type']
gpu_layers = variables['gpu_layers']
use_internet = variables['use_internet']


###FROM INTERNET 
if use_internet ==0:
    
    #load the local version of llama2
    llm = AutoModelForCausalLM.from_pretrained(llm_path
                                            , model_file=model_file
                                            , model_type=model_type
                                            , gpu_layers=gpu_layers)

    #sample prompt
    print(llm(instructions))
else: 
    #download and load the gguf version of llama2
    llm = AutoModelForCausalLM.from_pretrained("TheBloke/llama2_7b_merge_orcafamily-GGUF"
                                            , model_file="llama2_7b_merge_orcafamily.Q4_K_M.gguf"
                                            , model_type="llama", gpu_layers=50)

    #sample prompt
    print(llm(instructions))

    
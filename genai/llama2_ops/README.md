
# Llama 2 Examples and generalized frameworks 

### This is Repository of local implementations based off llama-recipes from the following resources: 
- [llama-recipes](https://github.com/meta-llama/llama-recipes?tab=readme-ov-file)
- [llama-cpp datacamp](https://www.datacamp.com/tutorial/llama-cpp-tutorial)
- [llama_cpp offical documentation](https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.Llama)_
- [promptingguideai](https://www.promptingguide.ai)

### Location of Virtual Environment used (for zjc local use)
- Venv Creation Command: virtualenv -p /usr/bin/python3.11 /home/zjc1002/envs/llama2_env
- Activate: source /home/zjc1002/envs/llama2_env/bin/activate
- Deactivate: source /home/zjc1002/envs/llama2_env/bin/deactivate

#### Major Packages (reference requirments.txt for full list of packages installed in llama2_env)
- llama-cpp-python


### Llama architecture vs tranditional Transformer architecture
![alt text](image.png)

The main difference between the LLaMa architecture and the transformers’:

- **Pre-normalization (GPT3)**: used to improve the training stability by normalizing the input of each transformer sub-layer using the RMSNorm approach, instead of normalizing the output.
- **SwigGLU activation function (PaLM)**: the original non-linearity ReLU activation function is replaced by the SwiGLU activation function, which leads to performance improvements.
- **Rotary Embeddings (GPTNeao)**: the rotary positional embeddings (RoPE) was added at each layer of the network after removing the absolute positional embeddings.


### Important Concepts

- **Rows on Processor Elements (ROPE):** Objective is to distribute the rows of the model layers across multiple GPUS or other processing elements. This can help scale training by allowing it to take advantage of parallel processing capabilities.  
    - In the context of the Llama model, setting *split_mode to 2 (LLAMA_SPLIT_MODE_ROW) would mean using RopE scaling, i.e., splitting the rows of the model across multiple GPUs. This can be beneficial if you have a large model and multiple GPUs available, as it can speed up the training process.*
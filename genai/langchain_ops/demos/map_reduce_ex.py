from langchain import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import WebBaseLoader
from os.path import expanduser
from langchain.callbacks.manager import CallbackManager
from pathlib import Path
import bs4
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import LlamaCpp
from langchain.chains import LLMChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

repo_id = "shaowenchen/llama-2-7b-langchain-chat-gguf"
filename = 'llama-2-7b-langchain-chat.Q4_K.gguf'
repo_type = "model"
local_dir = "/home/zjc1002/Mounts/llms/llama-2-7b-langchain-chat-gguf"
local_dir_use_symlinks = False
modelpath = Path(local_dir, filename) 
model_path = expanduser(modelpath)

n_gpu_layers = 4 # Change this value based on your model and your GPU VRAM pool.
n_batch = 10    # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
n_ctx = 512     # Should be between 128 and 1024, consider the amount of VRAM in your GPU.
#LOAD SAMPLE DATA
#Define Sample Web Based Loader 
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)

#load and split sample docs 
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512
                                               , chunk_overlap=50
                                               
                                               )
splits = text_splitter.split_documents(docs)
print(len(splits))

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path=model_path
    , n_gpu_layers=n_gpu_layers
    , n_batch=n_batch
    , f16_kv=True  # MUST set to True, otherwise you will run into problem after a couple of calls
)

template="""
Compose a concise and a brief summary of the following text:
TEXT: `{text}`
"""

prompt = PromptTemplate(
    input_variables=['text'],
    template=template
)

#chain the results 
chain = load_summarize_chain(
    llm,
    chain_type='stuff',
    prompt=prompt,
    verbose=False
)

#generate condensed summaries (top3)
summary = [chain.run([_split]) for idx, _split in enumerate(splits) if idx<3]
  
#loop over and generate summaries 
for split in splits: 
    summary = chain.run([Document(page_content = split.page_content)])
    break 
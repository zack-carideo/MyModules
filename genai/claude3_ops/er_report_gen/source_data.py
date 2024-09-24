import sys, os, yaml, logging , anthropic 
import pandas as pd
from pathlib import Path    

#set up basic logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
logger =  logging.getLogger(__name__)

#get config 
_root = Path.cwd()
sys.path.append(_root.as_posix())
from doc_loaders.rss_scraper import parse_rss_feeds
from doc_loaders.pdf_reader import read_pdfs
from prompts.summarize import generate_summary_prompt, chunk_summarize_combine



#config path 
_cfg_path = Path(_root, "config.yaml").as_posix()

#load config 
if Path(_cfg_path).exists():
    with open(_cfg_path, 'r') as file:
        cfg = yaml.safe_load(file)
else:
    raise FileNotFoundError(f"Config file not found at {_cfg_path}")

#global settings
device = cfg['device'] 
output_dir = cfg['data']['output']['dir']

#anthropic settings(api hosted llm)
anthropic_api_key = cfg['llms']['anthropic']['api_key']
anthropic_model = cfg['llms']['anthropic']['model_name']
max_tokens = cfg['llms']['anthropic']['max_tokens']
temperature = cfg['llms']['anthropic']['temperature']
top_p = cfg['llms']['anthropic']['top_p']
system_p = cfg['llms']['anthropic']['system_p']

#summarization 
summary_system_p = cfg['llms']['anthropic']['summarization']['system_p']
summary_min_text_length = cfg['llms']['anthropic']['summarization']['min_text_length_2_summarize']
summary_chunk_size = cfg['llms']['anthropic']['summarization']['ntoken_chunk_size']
summary_model = cfg['llms']['anthropic']['summarization']['model_name']


#in house llm settings 
chunk_docs = cfg['data']['input']['chunk_docs']
llm_dir = cfg['llms']['dir']
semantic_chunker_llm_name = cfg['llms']['semantic_chunking_model_name']

#rss feed settings
rss_feeds = cfg['data']['input']['rss_feeds']
pdf_paths = cfg['data']['input']['pdf_files']
rss_outfilename = cfg['data']['output']['rss_outname']
pdf_outfilename = cfg['data']['output']['pdf_outname']

#anthropic API 
client = anthropic.Anthropic(api_key=anthropic_api_key)

#parse rss data and save to file 
rss_data_path = parse_rss_feeds(rss_feeds
    , chunk_docs = chunk_docs
    , llm_dir =  llm_dir
    , model_name = semantic_chunker_llm_name
    , device = device
    , out_path = f"{output_dir}/{rss_outfilename}" 
    )

#parse pdf data and save to file 
pdf_data_path = read_pdfs(pdf_paths
                    , out_path = f"{output_dir}/{pdf_outfilename}")

#report data 
#load pre-scraped report data 
pdf_data = pd.read_parquet(f"{output_dir}/{pdf_outfilename}")
pdf_data['final_text'] = pdf_data['pdf_text'].copy()


#loop over pdfs before passing to cluade to extract risks to summarize if needed 
for _idx, _pdf_row in pdf_data.iterrows(): 

    #if pdf has more than 40 pages, generate a summary prompt
    if len(_pdf_row['pdf_text'])>summary_min_text_length: 

        logger.info(f"Summarizing pdf {_pdf_row['pdf_path']}")

        _summary  = chunk_summarize_combine( 
                    client
                , summary_model
                , _pdf_row['pdf_text']
                , chunk_size_tokens = summary_chunk_size
                , max_tokens = 4096
                , temperature = temperature
                , system = summary_system_p
                )

        pdf_data.loc[_idx,'final_text'] = _summary 
    
    else: 
        pdf_data.loc[_idx,'final_text'] = _pdf_row['pdf_text']
    
    #saved summarized data 
    pdf_data.to_parquet(f"{output_dir}/pdf_data_summary.pq")


#GET THE RSS DATA AND ADD AS OBSERVATION TO FINAL DF USED BY CLAUDE
#combine info from rss feeds and summarize 
rss_data = pd.read_parquet(f"{output_dir}/{rss_outfilename}")
rss_data = rss_data.drop_duplicates(subset='content',keep='first')
rss_text = '\n\n'.join(rss_data['content'].tolist())

# Example dictionary to append as a new row
new_row = {
    'pdf_path': f"{output_dir}/{rss_outfilename}"
    , 'pdf_pages': 100
    , 'pdf_text': rss_text
    , 'final_text': rss_text 
}

# Convert the dictionary to a DataFrame and append it to the existing DataFrame
new_row_df = pd.DataFrame([new_row])
final_data = pd.concat([pdf_data, new_row_df], ignore_index=True)

# Save the updated DataFrame to a file
final_data.to_parquet(f"{output_dir}/final_data.pq")









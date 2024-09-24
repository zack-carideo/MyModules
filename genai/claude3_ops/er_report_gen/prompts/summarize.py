import os,sys,anthropic, logging, yaml
import pandas as pd 
from pathlib import Path 


#set up basic logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
logger =  logging.getLogger(__name__)

_root = Path.cwd()
sys.path.append(_root.as_posix())
from .prompt_utils import get_completion, create_anthropic_prompt


def chunk_summarize_combine(
         client: anthropic.Anthropic
        , model_name:str
        , full_text:str
        , chunk_size_tokens:int = 20000
        , max_tokens:int = 4096
        , temperature:float = .1
        , system:str = None):

    """
    Chunk a large text document into smaller pieces and summarize each chunk.
    Combine the summaries into a single document.
    WARNING: I USE Anthropics API TO SUMMARIZE TEXT. IT COSTS MONEY!!!

    Args:
        full_text (str): Full text document to summarize.
        chunk_size_tokens (int): Number of tokens to include in each chunk.
        model_name (str): Name of the model to use for summarization.
        max_tokens (int): Maximum number of tokens to use in the summary.
        temperature (float): Temperature setting for the model.
        system (str): System setting for the model.
    
    Returns:
        str: Summarized text document.
    """
    
    #generate chunks of text
    chunks = [full_text[i:i+chunk_size_tokens] for i in range(0, len(full_text), chunk_size_tokens)]

    #summarize each chunk
    _summaries = []
    for idx,_chunk in enumerate(chunks):
    
        _summary = generate_summary_prompt(
                    client
                    , model_name
                    , _chunk
                    , temperature = temperature
                    , system = system
                    )

        _summaries.append(_summary)
        logger.info(f"completed summary for chunk {idx}")

    return '\n\n'.join(_summaries)


def generate_summary_prompt(client
                , model_name:str 
                , report_text: str
                , max_tokens: int = 4096
                , temperature: float = .1
                , system: str = None):

    """
    Generate a prompt for identifying risks and creating controls from a financial industry report.
    """

    #generate task specific prompt
    _prompt = fin_report_summary_prompt(report_text)

    #pass prompt to anthropic and get response 
    return get_completion(client
                    , model_name
                    , _prompt
                    , max_tokens=max_tokens
                    , temperature = temperature
                    , system = system 
                    , _type='text')

def fin_report_summary_prompt(report_text):

    _prompt = f"""
    Summarize the following financial industry report: 

        <report>
        [{report_text}]
        </report>
    
    The summary should be robust, clear, and precise.  Below are important themes of information to consider in summarizing the report: 
        1. Potential vulnerabilities to the economy (inflation, interest rates, market bubbles, etc.)
        2. Future risks regulators and financial insututions need to prepare for (climate change, digital transformation, cyber security, artifical intelligence,  etc.)
        3. How institutions and regulators are assessing financial risks (fair lending, consumer protection, stress tests,  etc.)
        4. How financial institutions can proactivly prepare for new emerging risks to the financial system (regulatory changes, technology advancements, control testing,  etc.)
        5. Assessing vulnerabilities to the financial system, and how those vulnerabilities might interact to amplify stress in the financial system(interaction between inflation and consumer spending may drive draw down of consumer savings, etc.)
        6. Controls and best practices for banks to implement to prevent risks from materializing.
 
    You can generate additional themes to use in your summary. Only include themes that are specific to the report. Use the maximum number of tokens to generate the summary. 

    Output your results in the following format:
    
    <summary>
    [Theme 1]:
        1. [bullet point 1]
        2. [bullet point 2]
        ...
        5. [bullet point n]

    [Theme 2]:
        1. [bullet point 1]
        2. [bullet point 2]
        ...
        5. [bullet point n]
    
    [Continue for all identified Risk Themes]
    ...
    </summary>

    """

    return _prompt


if __name__ =='__main__':

    #get config 
    _root = Path.cwd()
    sys.path.append(_root.parent.as_posix())

    #config path 
    _cfg_path = Path(_root.parent.as_posix(), "config.yaml").as_posix()

    #load config 
    if Path(_cfg_path).exists():
        with open(_cfg_path, 'r') as file:
            cfg = yaml.safe_load(file)
    else:
        raise FileNotFoundError(f"Config file not found at {_cfg_path}")

    #config params
    anthropic_api_key = cfg['llms']['anthropic']['api_key']
    anthropic_model = cfg['llms']['anthropic']['model_name']
    temperature = cfg['llms']['anthropic']['temperature']
    summary_system_p = cfg['llms']['anthropic']['summarization']['system_p']
    summary_min_text_length = cfg['llms']['anthropic']['summarization']['min_text_length_2_summarize']
    summary_chunk_size = cfg['llms']['anthropic']['summarization']['ntoken_chunk_size']
    summary_model = cfg['llms']['anthropic']['summarization']['model_name']

    device = cfg['device']
    output_dir = cfg['data']['output']['dir']
    pdf_outfilename = cfg['data']['output']['pdf_outname']

    #anthropic API 
    client = anthropic.Anthropic(api_key=anthropic_api_key)

    #load pre-scraped report data 
    pdf_data = pd.read_parquet(f"{output_dir}/{pdf_outfilename}")
    pdf_data['final_text'] = pdf_data['pdf_text'].copy()

    #loop over pdfs before passing to cluade to extract risks to summarize if needed 
    for _idx, _pdf_row in pdf_data.iterrows(): 

        #if pdf has more than 40 pages, generate a summary prompt
        if len(_pdf_row['pdf_text'])>summary_min_text_length: 

            logger.info(f"Summarizing pdf {_pdf_row['pdf_title']}")

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

    #save summarized data
    pdf_data.to_parquet(f"{output_dir}/pdf_data_summary.pq")
    logger.info(f"Summary of pdf data saved to {output_dir}/pdf_data_summary.pq")




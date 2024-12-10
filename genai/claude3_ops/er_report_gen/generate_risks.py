import os,sys,anthropic
from pathlib import Path 
import pandas as pd 
import yaml

_root = Path.cwd()
sys.path.append(_root.as_posix())
from prompt_utils import get_completion, create_anthropic_prompt



    df_path = "/home/zjc1002/Mounts/data/claude_outputs/final_data.pq"
    df = pd.read_parquet(df_path)


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
    
    device = cfg['device']
    output_dir = cfg['data']['output']['dir']
    pdf_outfilename = cfg['data']['output']['pdf_outname']

    #anthropic API 
    client = anthropic.Anthropic(api_key=anthropic_api_key)

    #risk and controls 
    for idx, row in df.iterrows(): 
        
        _response = generate_risk_controls_prompt(client
                        , anthropic_model
                        , row['final_text']
                        , max_tokens = 4096
                        , temperature = .1
                        , system = None)

        df.loc[idx, 'risk_controls'] = _response

    df.to_parquet(f"{output_dir}/step1_risk_controls.pq")


    #####
    ####START HERE
    #####
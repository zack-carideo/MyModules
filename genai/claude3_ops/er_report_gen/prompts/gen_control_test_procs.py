import os,sys,anthropic
from pathlib import Path 
import pandas as pd 
import yaml

_root = Path.cwd()
sys.path.append(_root.as_posix())
from prompt_utils import get_completion




#create prompt from url markdown output 
def control_testing_prompt(REGULATORY_RISK_CONTROLS
                   , RAW_REPORTS
                   ):
    _prompt = f"""
    You are an AI assistant tasked with generating control testing procedures for a large financial institution. Your goal is to evaluate the impact of financial and regulatory controls on a bank. Follow these instructions carefully to complete the task:

    First, analyze the following financial risk report. The Risk Controls section contains information about the controls in scope for testing:

    <financial_risk_reports>
    {REGULATORY_RISK_CONTROLS}
    </financial_risk_reports>

    You can also reference information from these raw financial reports to help inform your control testing procedures:

    <raw_financial_reports>
    {RAW_REPORTS}
    </raw_financial_reports>

    Now, proceed with the following steps:

    1. Analyze the Risk Controls section of the financial risk reports. For each Risk, select the 3 controls that are most likely to mitigate the risk. Note that each bullet in the Risk Controls section is formatted as follows:
    - [Risk Name] ::: [Control Name] ::: [Control Description]

    2. For each Control selected in step 1, create a step-by-step control testing procedure to evaluate how effectively each control mitigates the risk. Use the control information, the information from the raw financial reports, and your domain knowledge to generate these procedures.

    3. Ensure that each Control Testing Procedure contains an ordered list of steps that can be used to evaluate and quantify the control effectiveness (e.g., [Step 1: <description of step>, Step 2: <description of step>,..., Step n: <description of step>]). 

    4. Make sure the Control Testing Procedures are measurable, diverse, and incorporate control-specific information from the [Control Description].

    5. Output your results in the following format:

    <control_test_procedures>
    - [Risk Name] ::: [Control Name] ::: [Control Description] ::: [Control Testing Procedure]
    [Continue for all selected controls]
    </control_test_procedures>

    Additional guidelines:
    - Ensure that your output follows the specified format and includes all required information for each step.
    - Your response must contain valid control testing procedures for all selected controls.
    - Control testing procedures must be concise enough to fit testing procedures for all selected controls within the maximum token limit.
    - Use the maximum number of available tokens to provide thorough and concise results.
    - Do not include any explanations or comments outside of the specified output format.

    Begin your analysis and procedure generation now.
    """
    return _prompt



def generate_control_test_docs(client
                , model_name:str 
                , risk_controls: str
                , raw_report: str 
                , max_tokens: int = 4096
                , temperature: float = .1
                , system: str = None):

    """
    Generate a prompt for identifying risks and creating controls from a financial industry report.
    """

    #generate task specific prompt
    _prompt = control_testing_prompt(risk_controls, raw_report)

    #pass prompt to anthropic and get response 
    return get_completion(client
                    , model_name
                    , _prompt
                    , max_tokens=max_tokens
                    , temperature = temperature
                    , system = system 
                    , _type='text')


if __name__ == "__main__":
    
    df_path = "/home/zjc1002/Mounts/data/claude_outputs/step1_risk_controls.pq"
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
        
        _response = generate_control_test_docs(client
                        , anthropic_model
                        , row['risk_controls']
                        , row['final_text']
                        , max_tokens = 4096
                        , temperature = .1
                        , system = None)
        
        df.loc[idx, 'control_testing_procs'] = _response

    df.to_parquet(f"{output_dir}/step2_control_testing_procedures_v2.pq")
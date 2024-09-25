import os,sys,anthropic
from pathlib import Path 
import pandas as pd 
import yaml

_root = Path.cwd()
sys.path.append(_root.as_posix())
from prompt_utils import get_completion


def risk_and_controls_prompt(REPORT_TEXT):
    _prompt = f"""You are an AI researcher tasked with analyzing financial market publications to identify risks that have the potential to negatively impact the financial industry. Your specific task is to identify the latest emerging risks to Internal Audit teams in the financial industry for the year 2024 and create corresponding controls to mitigate these risks. Follow these instructions carefully to complete the task.

    First, review the following report carefully:

    <report>
    {{REPORT_TEXT}}
    </report>

    Now, proceed with the following steps:

    <Step1>
    1. Analyze the report to identify emerging risks that are expected to impact financial institutions in 2024. Focus on factors such as recency, potential impact, and specificity to the financial sector.

    2. For each risk you identify:
    a. Provide a concise name for the risk
    b. Write a brief description of the risk, explaining how it could affect financial institutions

    3. Ensure that each risk you identify is distinct and captures a unique emerging threat to financial institutions.

    4. Prioritize risks that are particularly relevant to Internal Audit teams in the financial industry.

    5. Aim to identify at least 3-5 significant emerging risks, but do not exceed 10 risks.

    6. Present your findings in the following format:

    <risks>
    - [Risk Name] ::: [Risk Description]
    - [Risk Name] ::: [Risk Description]
    (continue for each identified risk)
    </risks>

    Remember to focus on emerging risks specifically for the year 2024 and ensure that each risk is distinct and relevant to the financial sector. Your goal is to provide valuable insights that will help Internal Audit teams stay up-to-date on the newest potential threats to their institutions.
    </Step1>

    <Step2>
    For each risk identified in Step 1, your task is to create 5 controls to mitigate the risk. Each control should mitigate a unique aspect of the risk and must include details to satisfy all 5 components of the below FRASA Control Requirements.

    Here are the FRASA Control Requirements:
    <FRASA_CONTROL_REQUIREMENTS>
    1. Frequency: The frequency or timing of occurrence, how often will the control be evaluated
    2. Responsible Party: The party responsible for conducting the risk-mitigating activity (e.g., the director of trading reviews..., the accounting associate compares...)
    3. Activity: The specific risk-mitigating activity to be performed as part of the control (e.g., reconciliations are performed and reviewed between bank account balance and general ledger cash account balance and adjustments are recorded if needed)
    4. Source: The sources of information — The control should either define how management has addressed the completeness and accuracy of the information used in the control or there should be separate controls that address the completeness and accuracy of the information
    5. Action Taken: The action taken with the results of the control activity (for example, adjustments are made to the general ledger cash accounts, if needed, based on reconciliation to the bank balances)
    </FRASA_CONTROL_REQUIREMENTS>

    Use the risks from Step 1, the information from the report text, and all other knowledge available to you to create risk-specific controls. Ensure that each control description is comprehensive and addresses all five FRASA Control Requirements.

    Present your results in the following format:
    <Risk_Controls>
    - [Risk 1 Name] ::: [Control 1 Name] ::: [Detailed FRASA-compliant control description]
    - [Risk 1 Name] ::: [Control 2 Name] ::: [Detailed FRASA-compliant control description]
    .....
    - [Risk 1 Name] ::: [Control 5 Name] ::: [Detailed FRASA-compliant control description]
    - [Risk 2 Name] ::: [Control 1 Name] ::: [Detailed FRASA-compliant control description]
    - [Risk 2 Name] ::: [Control 2 Name] ::: [Detailed FRASA-compliant control description]
    .....
    - [Risk 2 Name] ::: [Control 5 Name] : [Detailed FRASA-compliant control description]
    ....
    [Continue for all risks and controls]
    </Risk_Controls>
    </Step2>

    After completing both steps, review your work to ensure all risks and controls are relevant, distinct, and properly formatted. Your final output should provide a comprehensive analysis of emerging risks for 2024 and detailed, FRASA-compliant controls to mitigate these risks, tailored specifically for Internal Audit teams in the financial industry.
    Use all available tokens to create detailed and specific controls for each risk.
    """

    return _prompt 


    
def generate_risk_controls_prompt(client
                , model_name:str 
                , report_text: str
                , max_tokens: int = 4096
                , temperature: float = .1
                , system: str = None):

    """
    Generate a prompt for identifying risks and creating controls from a financial industry report.
    """

    #generate task specific prompt
    _prompt = risk_and_controls_prompt(report_text)

    #pass prompt to anthropic and get response 
    return get_completion(client
                    , model_name
                    , _prompt
                    , max_tokens=max_tokens
                    , temperature = temperature
                    , system = system 
                    , _type='text')


if __name__ == "__main__":

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


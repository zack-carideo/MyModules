import os,sys,anthropic
from pathlib import Path 
import pandas as pd 
import yaml

_root = Path.cwd()
sys.path.append(_root.as_posix())
from prompt_utils import get_completion, create_anthropic_prompt




def risk_and_controls_prompt(report_text: str):


    hl_objective = f"""
    Your task is to analyze financial industry reports, identify risks from the report, and create FRASA compliant controls to mitigate each risk.
    First you will be provided a report to analyze. Review this information carefully. :  
    
    <report>
    [{report_text}]
    </report>

    Now, Follow the below instructions carefully to complete the task step by step."""

    task_1 = f"""
    <Step1>
    Create a bulleted overview of risks from the <report> that have the potential to negativly impact the financial sector, specifically Globally Systemic Important Banks(G-SIB).
    Each risk should be distinct and capture a unique emerging threat to financial institutions.
    Use the information from the <report> to identify the risks, and provide a description of the risk and how it could impact a G-SIB.
    If you cannot provide a consise description of the risk using the <report>, you can use your knowlodge of the banking industry to enrich the risk descriptions.
    The goal of this overview is to ensure Internal Audit Teams stays up to date on the newest risks that are expected to impact the financial institution in in the coming quarters(June 2024-June 2025)
    
    Output your results in the following format:

    <risks>
        - [Risk Name] ::: [Risk Description]
        - [Risk Name] ::: [Risk Description]
        ....
        [Continue for all risks identified]
    </risks>
    </Step1>\n
    """


    task_2 = f"""
    <Step2>
    For each risk identified in <Step 1>, your task is to create 5 controls to mitigate the risk. 
    Each control must adhere to the below <FRASA CONTROL REQUIRMENTS> and should mitigate a unique aspect of the risk
    
    Here are the FRASA Control Requirements:
        <FRASA CONTROL REQURIMENTS>
            1. Frequency: The frequency or timing of occurrence, how often will the control be evaluated
            2. Responsible Party: The party responsible for conducting the risk-mitigating activity (e.g., the director of trading reviews..., the accounting associate compares...)
            3. Activity: The specific risk-mitigating activity to be performed as part of the control (e.g., reconciliations are performed and reviewed between bank account balance and general ledger cash account balance and adjustments are recorded if needed)
            4. Source: The sources of information (if applicable) — The control should either define how management has addressed the completeness and accuracy of the information used in the control or there should be separate controls that address the completeness and accuracy of the information
            5. Action Taken: The action taken with the results of the control activity (for example, adjustments are made to the general ledger cash accounts, if needed, based on reconciliation to the bank balances)
        </FRASA CONTROL REQURIMENTS>

    Use the <risks> from <Step1> and the information from the <report_text> and all other knowlodge available to you to create risk specific controls. 
    Ensure that each control description is comprehensive and addresses all five FRASA Control Requirements.
    
    Present your results in the following format:
        <Risk Controls>
            - [Risk 1 Name] : [Control 1 Name] : [Detailed FRASA-compliant control description]
            - [Risk 1 Name] : [Control 2 Name] : [Detailed FRASA-compliant control description]
            .....
            - [Risk 1 Name] : [Control 5 Name] : [Detailed FRASA-compliant control description]
            - [Risk 2 Name] : [Control 1 Name] : [Detailed FRASA-compliant control description]
            - [Risk 2 Name] : [Control 2 Name] : [Detailed FRASA-compliant control description]
            .....
            - [Risk 2 Name] : [Control 5 Name] : [Detailed FRASA-compliant control description]
            ....
            [Continue for all risks and controls]
        </Risk Controls>
    </Step2>"""

    closing_text = f"""
    Ensure that your analysis is thorough, clear, and actionable for Internal Audit teams within financial institutions. 
    Use all available tokens to create detailed and specific controls for each risk.
    """

    _prompt = f"{hl_objective}\n{task_1}\n{task_2}\n{closing_text}"

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


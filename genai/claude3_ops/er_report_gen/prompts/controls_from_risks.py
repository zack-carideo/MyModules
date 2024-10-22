import os,sys,anthropic, logging
from pathlib import Path 
import pandas as pd 
import yaml

_root = Path.cwd()
sys.path.append(_root.as_posix())
from prompt_utils import get_completion

#set up basic logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
logger =  logging.getLogger(__name__)



#create prompt from url markdown output 
def controls_from_risks(RISKS):

    _prompt = f"""
    You are an AI internal auditor tasked with identifying risks to the financial industry and creating controls to mitigate these risks. Your specific task is to analyze a set of risks identified by financial regulators and create objective, measurable controls to mitigate them. Follow these instructions carefully to complete the task:

    First, review the following list of risks:

    <risks>
    {RISKS}
    </risks>

    Next, familiarize yourself with the FRASA Control Requirements. Each control you generate must satisfy all of these requirements:

    <FRASA_Control_Requirements>
    1. Frequency: The frequency or timing of occurrence, how often will the control be evaluated
    2. Responsible Party: The party responsible for conducting the risk-mitigating activity
    3. Activity: The specific risk-mitigating activity to be performed as part of the control
    4. Source: The sources of information used in the control
    5. Action Taken: The action taken with the results of the control activity
    </FRASA_Control_Requirements>

    Now, proceed with the following steps:

    1. For each risk in the <risks> list, identify 5 unique emerging sub-risks that capture different components of the main risk. Use the most current information available to you to ensure these sub-risks are relevant for 2024.

    2. For each sub-risk, generate a control to mitigate it. Ensure that each control:
    a. Is objective and measurable
    b. Complies with all five FRASA Control Requirements
    c. Is unique and distinct from the other controls generated for the same risk

    3. Present your results in the following format:

    <risk_controls>
    - [Risk 1 Name] ::: [Control 1 Name] ::: [Detailed FRASA-compliant control description]
    - [Risk 1 Name] ::: [Control 2 Name] ::: [Detailed FRASA-compliant control description]
    ...
    - [Risk 1 Name] ::: [Control 5 Name] ::: [Detailed FRASA-compliant control description]
    - [Risk 2 Name] ::: [Control 1 Name] ::: [Detailed FRASA-compliant control description]
    - [Risk 2 Name] ::: [Control 2 Name] ::: [Detailed FRASA-compliant control description]
    ...
    - [Risk 2 Name] ::: [Control 5 Name] ::: [Detailed FRASA-compliant control description]
    ...
    [Continue for all risks and controls]
    </risk_controls>

    Ensure that each control description is comprehensive and addresses all five FRASA Control Requirements. Use all available tokens to create detailed and specific controls for each risk.

    After completing these steps, review your work to ensure all risks and controls are relevant, distinct, and properly formatted. Verify that all controls satisfy the FRASA Control Requirements.

    Your final output should provide a comprehensive analysis of emerging risks for 2024 and detailed, FRASA-compliant controls to mitigate these risks, tailored specifically for Internal Audit teams in the financial industry.
    """

    return _prompt

#You will use the FRASA compliant control descriptions to inform each control testing procedure. the risks and controls provided will be formatted as follows:
        
def control_testing_procedures_from_controls(controls):
    _prompt = f"""
    You are an AI internal auditor tasked with generating control testing procedures for a large financial institution. 
    Your goal is to evaluate the impact of financial and regulatory controls on a bank to help them mitigate risk. 
    Follow these instructions carefully to complete the task:

    First, analyze the following summary of controls. The Risk Controls section of the risk summary contains information about the controls in scope for testing:

    <report_summary>
    {controls}
    </report_summary>

    Now, proceed with the following steps:

    1. Analyze the Risk Controls section of the report summary. Note that each bullet in the Risk Controls section is formatted as follows:
    - [Risk Name] ::: [Control Name] ::: [Control Description]

    2. For each Control in step 1, create a step-by-step control testing procedure to evaluate how effectively each control mitigates the risk. Use the control information and your domain knowledge to generate these procedures.

    3. Ensure that each Control Testing Procedure contains an ordered list of steps that can be used to evaluate and quantify the control effectiveness (e.g., [Step 1: <description of step>, Step 2: <description of step>,..., Step n: <description of step>]). 

    4. Make sure the Control Testing Procedures are measurable, diverse, and incorporate control-specific information from the [Control Description].

    5. Output your results in the following format:

    <control_test_procedures>
    - [Risk Name] ::: [Control Name] ::: [Control Testing Procedure]
    [Continue for all controls in the risk summary]
    </control_test_procedures>

    Additional guidelines:
    - Ensure that your output follows the specified format and includes all required information for each step.
    - Your response should contain valid control testing procedures for all controls listed in the risk summary.
    - Control testing procedures must be concise enough to fit testing procedures for all controls within the maximum token limit.
    - Use the maximum number of available tokens to provide thorough and concise results.
    - Do not include any explanations or comments outside of the specified output format.

    Begin your analysis and procedure generation now.
        """
    return _prompt


def cot_controls_and_testing_procs(client
                , model_name:str 
                , risks: list
                , max_tokens: int = 4096
                , temperature: float = .1
                , system: str = None):
    
    """
    Generate a prompt for identifying risks and creating controls from a financial industry report.
    Generate a prompt for control testing procedures from the controls generated in the first prompt.
    """
    #generate task specific prompt
    _prompt = controls_from_risks(risks)

    #pass prompt to anthropic and get response 
    _controls = get_completion(client
                    , model_name
                    , _prompt
                    , max_tokens=max_tokens
                    , temperature = temperature
                    , system = system 
                    , _type='text')

    logger.info(f"controls for {risks}")
    logger.info(print(_controls))

    #generate task specific prompt
    _prompt2 = control_testing_procedures_from_controls(_controls)

    #return final response 
    _procedures =  get_completion(client
                            , model_name
                            , _prompt2
                            , max_tokens=max_tokens
                            , temperature = temperature
                            , system = system 
                            , _type='text')
    
    return _controls, _procedures




if __name__ == "__main__":
    
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
    risks = cfg['analysis']['risk_themes']

    #loop over each risk, generate controls, and then testing procedures (COT)
    output_ctps = []
    for idx,_risk in enumerate(risks): 
        
        _control, _procedure = cot_controls_and_testing_procs(client
                        , anthropic_model
                        ,[_risk]
                        , max_tokens = 4096
                        , temperature = .1
                        , system = None)

        output_ctps.append({'risk':_risk,'controls': _control, 'control_test_procedures':_procedure})

    #save it out 
    pd.DataFrame(output_ctps).to_parquet(f"{output_dir}/matt_risks_controls_and_testing_procedures_9_24_24.pq")


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
    You are an internal auditor working for a large financial institution. 
    Your task is to create controls to mitigate a set of risks identified by financial leaders. 
    The controls must be objective and measurable. All controls must satisfy the FRASA Control Requirements.Controls should be written in complete sentences. 
        
    Here are the FRASA Control Requirements:
    1. Frequency: The frequency or timing of occurrence, how often will the control be evaluated
    2. Responsible Party: The party responsible for conducting the risk-mitigating activity (e.g., the director of trading reviews..., the accounting associate compares...)
    3. Activity: The specific risk-mitigating activity to be performed as part of the control (e.g., reconciliations are performed and reviewed between bank account balance and general ledger cash account balance and adjustments are recorded if needed)
    4. Source: The sources of information (if applicable) — The control should either define how management has addressed the completeness and accuracy of the information used in the control or there should be separate controls that address the completeness and accuracy of the information\n5. Action Taken: The action taken with the results of the control activity (for example, adjustments are made to the general ledger cash accounts, if needed, based on reconciliation to the bank balances)

    You will be provided with a list of risks in the financial industry. Your task is to generate 5 controls for each risk that comply with the FRASA Control Requirements.
    Here is the list of risks:
    <risks>
    {RISKS}
    </risks>
    
    Follow the below steps to complete the task, think step by step :

    <Step1>

        1. For each risk in the list, generate 5 controls that can be used to mitigate the risk.
        2. Ensure that each control complies with all five FRASA Control Requirements.
        3. Consider factors such as recency, authority of the source, and specificity to the financial sector when creating controls.
        4. Use all information available to you to generate these controls.
        5. Present your results in the following format:
    
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

    </Step 1>

    Ensure that each control description is comprehensive and addresses all five FRASA Control Requirements. Use all available tokens to create detailed and specific controls for each risk."""
                
    return _prompt

#You will use the FRASA compliant control descriptions to inform each control testing procedure. the risks and controls provided will be formatted as follows:
        
def control_testing_procedures_from_controls(controls):
    _prompt = f"""
        You are a financial industry regulator tasked with generating control testing procedures given a set of risks and associated FRASA compliant Controls. 
        Specificaly you will be given a financial risk report with risks and associated FRASA compliant controls. 
        You will create step by step testing procedures for each control to evaluate the financial and regulatory impact of the control on a bank. 
        Use the information from the Risk Controls(delimited as <Risk Controls></Risk Controls>) section of the financial risk report to define the controls in scope of control testing.
        For refernce , the <Risk Controls> Section of the financial risk report is formatted as follows: - [Risk Name] : [Control Name] : [FRASA-compliant control description]

        The financial risk report with the controls requiring contorl testing procedures is below:
        <financial_risk_report>
        [{controls}]
        </financial_risk_report>


        Follow these instructions carefully to complete the task:

        <Step1> 
        Control Testing Procedures: 

        a. For each Control from the <financial_risk_report> create a step by step control testing procedure to evalute how effectivly each control mitigates the risk. 
            Each Control Testing Procedure should contain an ordered list of steps that can be used to evaluate the control(ex. [Step 1: <description of step>, Step 2: <description of step>,..., Step n: <description of step> ]). 
            Control Testing Procedures should be measurable, diverse, and incorporate control specific information from the [FRASA-compliant control description].
            You can also use your own knowlodge of financial control testing to generate the control testing procedures. 
            Control testing procedures must be consise enough to fit testing procedures for all controls in the <financial_risk_report> within the maximum token limit.

        b. Output your results in the following format.(NOTE: The [Risk Name], [Control Name], and [FRASA-compliant control Description] should come directly from the <financial_risk_report>):

            <control_test_procedures>
             - [Risk Name] : [Control Name] : [FRASA-compliant control Description] : [Control Testing Procedure]
            ....

            [Continue for all selected controls]
            ...
            </control_test_procedures>
        </Step1>    

        Ensure that your output follows the specified formats and includes all required information for each step. 
        Your response must contain valid control testing procedures for all selected controls. Use the maximum number of avilable tokens to provide through and concise results.                 
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

    logger.info(f"controls for {_risk}")
    logger.info(print(_controls))

    #generate task specific prompt
    _prompt2 = control_testing_procedures_from_controls(_controls)

    #return final response 
    return get_completion(client
                            , model_name
                            , _prompt2
                            , max_tokens=max_tokens
                            , temperature = temperature
                            , system = system 
                            , _type='text')




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

    output_ctps = []
    for idx,_risk in enumerate(risks): 
        
        _response = cot_controls_and_testing_procs(client
                        , anthropic_model
                        ,[_risk]
                        , max_tokens = 4096
                        , temperature = .1
                        , system = None)

        output_ctps.append({'risk':_risk,'control_test_procedures':_response})

    

    df.to_parquet(f"{output_dir}/step2_control_testing_procedures.pq")


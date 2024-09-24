import os,sys,anthropic
from pathlib import Path 
import pandas as pd 
import yaml

_root = Path.cwd()
sys.path.append(_root.as_posix())
from prompt_utils import get_completion, create_anthropic_prompt,extract_text_between_anchors




#create prompt from url markdown output 
def control_testing_prompt(REGULATORY_RISK_CONTROLS
                   , RAW_REPORTS
                   ):
    _prompt = f"""
        You are a financial industry regulator tasked with generating control testing procedures. Specificaly you will be asked to create step by step testing procedures to evaluate the impact of financial and regulatory controls on a bank. 
        You will analyze the following financial risk report. Use the information from the Risk Controls(delimited as <Risk Controls></Risk Controls>) section of the report to define the controls in scope of control testing.:

        <financial_risk_reports>
        [{REGULATORY_RISK_CONTROLS}]
        </financial_risk_reports>

        you can also reference information from these raw_financial_reports to help inform your control testing procedures. 
        <raw_financial_reports>
        [{RAW_REPORTS}]
        </raw_financial_reports>


        Follow these instructions carefully to complete the task:

        <Step1>
        Identify Most Relevant Controls: 
        a. For each Risk Theme in the  <Risk Controls> section of <financial_risk_reports>, select the 2 controls that are most likely to mitigate the risk.For Reference , Each bullet in the <Risk Controls> section of the <financial_risk_reports> is formatted as follows:
            - [Risk Theme]: [Control Name] : [Control Description] 
        
        b. Do not  output the results from this step. the information will be used in the next step.

        </Step1>
        
        <Step2> 
        Control Testing: 

        a. For each Control selected in <Step1> create a step by step control testing procedure to evalute how effectivly each control mitigates the risk. 
            Each Control Testing Procedure should contain an ordered list of steps that can be used to evaluate the control(ex. [Step 1: <description of step>, Step 2: <description of step>,..., Step n: <description of step> ]). 
            Control Testing Procedures should be measurable, diverse, and incorporate control specific information from the [Control Description].
            Use the control information from <Step1> and the information from the <raw_financial_reports> to generating control testing procedures. 
            You can also use your own knowlodge of financial control testing. Control testing procedures must be consise enough to fit testing procedures for all selected control from <Step1> within the maximum token limit.

        b. Output your results in the following format:
            <control_test_procedures>
             - [Risk Theme] : [Control Name] : [Control Description] : [Control Testing Procedure]
            ....

            [Continue for all selected controls]
            ...
            </control_test_procedures>
        </Step2>    

        Ensure that your output follows the specified formats and includes all required information for each step. 
        Your response must contain valid control testing procedures for all selected controls. Use the maximum number of avilable tokens to provide through and concise results.                 
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
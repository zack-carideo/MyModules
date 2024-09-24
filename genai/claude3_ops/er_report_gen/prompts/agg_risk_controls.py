import os,sys,anthropic
from pathlib import Path 
import pandas as pd 
import yaml

_root = Path.cwd()
sys.path.append(_root.as_posix())
from prompt_utils import get_completion, create_anthropic_prompt

#create prompt from url markdown output 
def create_prompt2(REGULATORY_DOCUMENTS
                  , type='text'):
    return {
            "role": "user",
            "content": [
                {
                    "type": type,
                    "text": f"""
        You are a financial industry regulator tasked with grouping financial risks from regulatory reports into risk themes and identifying controls and control testing procedures to mitigate these risks. You will analyze the following list of financial risk reports:

        <regulatory_documents>
        {{{REGULATORY_DOCUMENTS}}}
        </regulatory_documents>

        Follow these instructions carefully to complete the task:
        
        <Step1> 
        Risk Theme Mapping:
        a. Review all <risks> sections in the regulatory documents.
        b. Group the <risks> into no more than 10 Risk Themes.
        c. Map <risks> from the original documents to a Risk Theme. Ensure that each <risk> is mapped to only one Risk Theme.
        d. Output your results in the following format:
            
            <risk_themes>
            [Risk Theme 1]: [<risks>list of mapped risks</risks>]
            [Risk Theme 2]: [<risks>list of mapped risks</risks>]
            [Continue for all identified Risk Themes]
            ...
            </risk_themes>
        </Step1>

        <Step2>
        Risk Summary: 
        a. For each Risk Theme identified in step 1, provide a summary of how the mapped <risks> impact the overall Risk Theme, and how the overall Risk Theme impacts the financial industry.
        b. Output your results in the following format:
        
            <risk_summary>
            [Risk Theme 1]: [Risk Summary Overview]
            [Risk Theme 2]: [Risk Summary Overview]

            [Continue for all identified Risk Themes]
            ...
            </risk_summary>
        </Step2>

        <Step3>
        Key Controls:
        a. For each Risk Theme, identify 5 controls that can mitigate the <risks> associated with the Risk Theme. Use the <risk_controls> sections from all regulatory_documents as a reference. 
        b. Output your results in the following format:
            <key_controls>
            [Risk Theme 1]:
                1. [Control 1]
                2. [Control 2]
                3. [Control 3]
                4. [Control 4]
                5. [Control 5]

            [Risk Theme 2]:
                1. [Control 1]
                2. [Control 2]
                3. [Control 3]
                4. [Control 4]
                5. [Control 5]

            [Continue for all identified Risk Themes]
            ...
            </key_controls>
        </Step3>    

Ensure that your output follows the specified formats and includes all required information for each step. Your response should use as many tokens as required to provide through and concise results.                 
"""
                }
            ]
        }


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
        Control Testing: 
        a. For each Control in the  <Risk Controls> section of <financial_risk_reports>, create a step by step control testing procedure 
        to evalute how well each control mitigates the risk. Each bullet in the <Risk Controls> section of the <financial_risk_reports> is formatted as follows:
            - [Risk Theme]: [Control Name] : [Control Description] 
        
        
        Control Testing Procedures should be measurable, diverse, and incorporate control specific information from the [Control Description].  
        Use the <Risk Controls> information from the <financial_risk_reports> and the information from the <raw_financial_reports> to generating control testing procedures. You can also use your own knowlodge of financial control testing.    
        b. Output your results in the following format:
            <control_test_procedures>
            
            [Risk Theme 1]:
                - [Control 1]
                    control testing Instructions:
                    1. [Step 1]
                    2. [Step 2]
                    3. [Step 3]
                    ...

            [Risk Theme 2]:
                - [Control 1]
                    control testing Instructions:
                    1. [Step 1]
                    2. [Step 2]
                    3. [Step 3]
                    ...
            ....

            [Continue for all controls for each risk theme]
            ...
            </control_test_procedures>
        </Step1>    

        Ensure that your output follows the specified formats and includes all required information for each step. Your response should use the maximum number of avilable tokens to provide through and concise results.                 
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
        break
        df.loc[idx, 'risk_controls'] = _response

    df.to_parquet(f"{output_dir}/step2_control_testing_procedures.pq")

    _prompt = control_testing_prompt(row['risk_controls'], row['final_text'])

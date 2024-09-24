import os,sys,anthropic
from pathlib import Path
#get config 
_root = Path.cwd()
sys.path.append(_root.as_posix())

#function to generate prompt template
def get_completion(client
                   , model_name 
                   , _text2analyze
                   , max_tokens=4096
                   , temperature = .9
                   , system= None
                   , _type='text'):
    
    if system is not None: 
        return client.messages.create(
            model=model_name,
            max_tokens=max_tokens,
            temperature = temperature, 
            system = system, 
            messages=[create_anthropic_prompt(_text2analyze, _type=_type)]
        ).content[0].text
    else: 
        return client.messages.create(
            model=model_name,
            max_tokens=max_tokens,
            temperature = temperature, 
            messages=[create_anthropic_prompt(_text2analyze, _type=_type)]
        ).content[0].text


#create prompt from url markdown output 
def create_anthropic_prompt(prompt_text: str=None 
, _type='text'
, _role='user'):

    return {
            "role": _role,
            "content": [
                {
                    "type": _type,
                    "text": prompt_text
                }
            ]
        }


#extract text between two anchors
def extract_text_between_anchors(_str:str
                                 , start_anchor:str
                                 , end_anchor:str
                                 , n_char_past_end:int = 1
                                 )->str:
    
    # Find the location of the first occurrence of a string pattern
    first_occurrence = _str.find(start_anchor)

    # Find the location of the second occurrence of a string pattern
    second_occurrence = _str.find(end_anchor, first_occurrence + n_char_past_end)
    return _str[first_occurrence:second_occurrence]

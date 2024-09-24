from pypdf import PdfReader
from typing import List
import pandas as pd

#read a pdf 
def read_pdf(pdf_path:str)-> str:

    reader = PdfReader(pdf_path)    
    number_of_pages = len(reader.pages)
    text = ''.join(page.extract_text() for page in reader.pages)  
    return {
         'pdf_path': pdf_path
        , 'pdf_pages': number_of_pages
        , 'pdf_text': text
        }


def read_pdfs(pdf_paths:List[str],out_path: str = None)-> List[str]:
    """
    Reads a list of pdfs and returns a list of dictionaries containing the pdf path, number of pages, and text.
    Args:
        - pdf_paths (List[str]): List of pdf file paths to read.
        - out_path (str, optional): Path to save the output DataFrame as a parquet file. Defaults to None.
    """

    _pdfs =  [read_pdf(pdf_path) for pdf_path in pdf_paths]

    #save 2 file or return df 
    if out_path:
        pd.DataFrame(_pdfs).to_parquet(out_path)
        return out_path 
    else: 
        return pd.DataFrame(_pdfs)

# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 17:15:28 2019

@author: zjc10
"""
import string
import re 
import bs4
import readability 
import unicodedata
import pandas as pd 
from datetime import datetime

import nltk

word_tokenizer = nltk.tokenize.word_tokenize 
sent_tokenizer = nltk.tokenize.sent_tokenize 


#basic word and sentence tokenization
def tokenize_sents(doc,tokenizer = sent_tokenizer): 
    return tokenizer(doc) 
   
def tokenize_words(sent,tokenizer = word_tokenizer):
    return tokenizer(sent)


#converting pandas df to json 
df_in_path = "C:\\Users\\zjc10\\Desktop\\Projects\\data\\news\\webhose_news\\webhose_df.pickle"
json_out_path = 'C:\\Users\\zjc10\\Desktop\\Projects\\data\\news\\webhose_news\\webhose_samp_elstc.json'

#convert dataframe to json
def df_to_json(df_in_path, json_out_path,vars2save = ['key','title']):
    """
    inputs:
        -df_in_path(path to file or existing df):"C:\\Users\\zjc10\\Desktop\\Projects\\data\\news\\webhose_news\\webhose_df.pickle"
        -json_out_path: 'C:\\Users\\zjc10\\Desktop\\Projects\\data\\news\\webhose_news\\webhose_samp_elstc.json'
        
    syntax ex:
        df_to_json(df_in_path, json_out_path,vars2save = ['key','title'])
    """
    #input data
    if type(df_in_path).__name__ == 'str':
        data = pd.read_pickle(df_in_path)
    else:
        data = df_in_path
        
    #convert dataframe to json 
    data[vars2save].head(2).to_json(json_out_path, orient='records')

    return json_out_path

#source data string cleaning funtions 
#handles cleaning of html sourced text 
def clean_text_str(text):
    text = text.replace("\n"," ").replace("\t"," ").replace("\r"," ")
    text = re.sub(r" +"," ",text)
    text = text.strip()
    return text 

def clean_unicode_text(text):
    str_test = clean_text_str(str(text))
    new_str =unicodedata.normalize("NFKD",str_test)
    return new_str 

def clean_html_text(text):
    out_str = clean_unicode_text(text)
    if bool(bs4.BeautifulSoup(out_str, "html.parser").find()):
        doc = readability.Document(str(out_str))
        soup = bs4.BeautifulSoup(doc.summary(),"lxml")
        article_text = " ".join(soup.findAll(text=True))
        return article_text 
    else:
        return out_str
    
#2nd level string cleaning 
#text processing for modeling (beyond basic input text formatting)
def clean_sentence(sentence,PUNCTUATION=string.punctuation+"\\\\",stemmer = None, lower=False,stopwords = None):
    sentence = sentence.encode('ascii',errors = 'ignore').decode()
    sentence=re.sub(f'[{PUNCTUATION}]',' ',sentence)
    sentence = re.sub(' {2,}',' ', sentence)
    if lower:
        sentence= sentence.lower().strip()
    else:
        sentence= sentence.strip()
    
    if stopwords: 
        sentence = ' '.join([word for word in sentence.split() if word not in stopwords])
    
    if stemmer: 
        sentence = ' '.join([stemmer.stem(word) for word in sentence.split()])
    return sentence 

#convert all null dates to 1970 
def safe_date(date_value):
    return (
        pd.to_datetime(date_value) if not pd.isna(date_value)
            else  datetime(1970,1,1,0,0)
    )
    
#convert blank string to safe valeus 
#syntax: df['Hold'] = df['PossiblyBlankField'].apply(safe_value)
def safe_value(field_val,default_val='default_nan'):
    return field_val if not pd.isna(field_val) else default_val


#extract gps location from string 
#
import re 
def compile_gps_pattern():
    lat = r'([-]?[0-9]?[0-9][.][0-9]{2,10})'
    lon = r'([-]?1?[0-9]?[0-9][.][0-9]{2,10})'
    sep = r'[,/ ]{1,3}'
    re_gps = re.compile(lat + sep + lon)
    return re_gps

def get_gps_inf(str_, re_pattern = compile_gps_pattern()):
    #ex.re_pattern.findall('abc is at 45.344, -121.9431')
    return re_pattern.findall(str_)


#get date nmonths out 
#def getDate_nMonthsOut(initial_date = None):
    



##FILTER OUT LOW INFORMATION WORDS 
##NEED TO GET THIS WORKING
def filterCommonWords(BOWCoprus,low_value_thresh = .02):
	from gensim.models import TfidfModel
	from gensim import models
	#filter out common words 
	#save copy of original corpus 
	CORPUS = list(BOWCoprus)

	#create td-idf model object using dictonary
	tfidf = models.TfidfModel(CORPUS, id2word = dictionary)

	#filter low value words
	low_value = low_value_thresh

	for i in range(0, len(CORPUS)):
	    bow = corpus[i]
	    low_value_words = [] #reinitialize to be safe. You can skip this.
	    low_value_words = [id for id, value in tfidf[bow] if value < low_value]
	    new_bow = [b for b in bow if b[0] not in low_value_words]

	    #reassign        
	    CORPUS[i] = new_bow
	    
	#length of each new corpus with stop words removed 
	Original = np.array(map(len,corpus)) 
	NoStopWrds = np.array(map(len,CORPUS))

	#difference in original string vs scrubbed string
	diff = Original-NoStopWrds
	print("No Stop Words Found in:", len(diff[diff==0]),"Strings")
	print("Stop Words removed from:",len(diff[diff>0]),"Strings")

	return CORPUS   
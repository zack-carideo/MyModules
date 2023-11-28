import pandas as pd 
import sys 
import os 
import os.path 
from os import path
import urllib
import fnmatch
import json 

###########
#CFPB DATA#
###########
#location to store cfpb data 
directory= "C:\\Users\\zjc10\\Desktop\\Projects\\data\\"
filename= "cfpb_complaints.csv"

#download cfpb dataset if not already on drive
if path.exists(directory+filename)==False:
    url = 'https://data.consumerfinance.gov/api/views/s6ew-h6mp/rows.csv?accessType=DOWNLOAD'
    urllib.request.urlretrieve(url,directory+filename)

###################
####WEBHOSE DATA###
###################
#cannot download on fly, must login to webhose first 
blog_dir = "C:\\Users\\zjc10\\Desktop\\Projects\\data\\news\\webhose_news\\webhose_news_json"
files = fnmatch.filter(os.listdir(blog_dir),'*.json')

#loop over each json file and concat to dataframe (very innefficent)
webhose_df = pd.DataFrame()
for i in range(0,len(files)):
    with open(blog_dir+'\\'+files[i],'r',encoding='cp866') as json_file:
        data = json.load(json_file)
        key = data['uuid']
        date = data['published']
        link = data['url']
        title = data['title']
        author = data['author']
        text = data['text']

    json2df = [{'key':key,'date':date,'title':title,'author':author,'link':link,'text':text}]
    webhose_df = pd.concat([webhose_df,pd.DataFrame(json2df)])    

#save output to pickle
webhose_df.to_pickle("C:\\Users\\zjc10\\Desktop\\Projects\\data\\news\\webhose_news\\webhose_df.pickle")

##########################
##GENERAL NEWS ARTICLES###
##########################
news_dir = "C:\\Users\\zjc10\\Desktop\\Projects\\data\\news\\general_news_scrape\\general_news_csvs"
files = fnmatch.filter(os.listdir(news_dir),'*.csv')
general_news_df = pd.DataFrame()

for i in range(0,len(files)):
    general_news_df = pd.concat([general_news_df ,pd.read_csv(news_dir+"\\"+files[i])])
general_news_df.to_pickle("C:\\Users\\zjc10\\Desktop\\Projects\\data\\news\\general_news_scrape\\general_news.pickle")
#document transformers are used to parse source data into langchain data

from langchain.document_loaders import AsyncChromiumLoader

#HTML 
from langchain.document_transformers import BeautifulSoupTransformer
import nest_asyncio
nest_asyncio.apply()

#HTML2Text
from langchain.document_loaders import AsyncHtmlLoader
from langchain.document_transformers import Html2TextTransformer

#inputs 
urls = ['https://www.ecu.edu']
tags2extract = ['p', 'li', 'div', 'a'] #for bs4 doc parser

# HTML
# beautiful soup (https://python.langchain.com/docs/integrations/document_transformers/beautiful_soup)
# When to use: you want to extract specific information and clean up the HTML content according to your needs.
# EX.Scrape text content within <p>, <li>, <div>, and <a> tags from the HTML content:

# 1.Load HTML
loader = AsyncChromiumLoader(urls)
html = loader.load()

# 2.Transform
bs_transformer = BeautifulSoupTransformer()
docs_transformed = bs_transformer.transform_documents(
    html, tags_to_extract=tags2extract
)


# HTML2Text 
# html2text (https://python.langchain.com/docs/integrations/document_transformers/html2text)
# when to use: you want to extract text content from HTML content.

# 1.load html
loader = AsyncHtmlLoader(urls)
docs = loader.load()

# 2.parse text from html 
html2text = Html2TextTransformer()
docs_transformed = html2text.transform_documents(docs)

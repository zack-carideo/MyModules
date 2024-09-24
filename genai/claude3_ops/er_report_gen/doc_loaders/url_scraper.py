from langchain_community.document_loaders import RecursiveUrlLoader
import re
from bs4 import BeautifulSoup
def bs4_extractor(html: str) -> str:
    soup = BeautifulSoup(html, "xml")
    return re.sub(r"\n\n+", "\n\n", soup.text).strip()

loader = RecursiveUrlLoader(
    "https://www.occ.gov/rss/occ_alerts.xml"
    , extractor=bs4_extractor
    # max_depth=2,
    # use_async=False,
    # extractor=None,
    # metadata_extractor=None,
    # exclude_dirs=(),
    # timeout=10,
    # check_response_status=True,
    # continue_on_failure=True,
    # prevent_outside=True,
    # base_url=None,
    # ...
)




docs = loader.load()
from web_tools.core.engines.google import Search as GoogleSearch

# init a search engine
gsearch = GoogleSearch(proxy=None)

# will automatically parse Google and corresponding web pages
gresults = gsearch.search('The robot is opening the refrigerator door.', cache=True, page_cache=True, topk=1, end_year=2024)
import ipdb;ipdb.set_trace()
print(gresults)
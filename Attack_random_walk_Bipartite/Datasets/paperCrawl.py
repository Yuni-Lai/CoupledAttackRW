import urllib.request
import feedparser
import time
import json
import os

def main():
    f = open("arxivData.json", "a")
    base_url = 'http://export.arxiv.org/api/query?';
    search_query = 'cat:cs.CV+OR+cat:cs.AI+OR+cat:math.AC+OR+cat:math.QA+OR+cat:nlin.CD+OR+cat:physics.optics+OR+cat:econ.EM'
    #'cat:cs.CV+OR+cat:cs.AI+OR+cat:cs.LG+OR+cat:cs.CL+OR+cat:cs.NE+OR+cat:stat.ML' # search for CV,CL,AI,LG,NE,ML field papers
    start = 0
    max_results = 1000
    count = 0
    fail_count = 0
    result_list = []
    #try:
    while start < 50000:
        time.sleep(5)
        query = 'search_query=%s&start=%i&max_results=%i' % (search_query,start,max_results)
        start = start + 1000
        feedparser._FeedParserMixin.namespaces['http://a9.com/-/spec/opensearch/1.1/'] = 'opensearch'
        feedparser._FeedParserMixin.namespaces['http://arxiv.org/schemas/atom'] = 'arxiv'
        response = urllib.request.urlopen(base_url+query).read()
        #response = urllib.request.urlopen(base_url).read()
        feed = feedparser.parse(response)
        for entry in feed.entries:
            #Arxiv id
            temp_sic = {}
            c1_data = str(entry.id.split('/abs/')[-1])
            #date
            #c2_data = str(entry.published)
            #year = int(str(c2_data[:4]))
            #month = int(str(c2_data[5:7]))
            #day = int(str(c2_data[8:10]))
            #Title
            c3_data = str(entry.title)
            #author
            c4_data = str(entry.authors)
            #Pdf link
            #c5_data = str(entry.links)
            #c6_data = str(entry.tags)
            #c7_data = str(entry.summary)
            temp_dic={
            "id" : c1_data,
            "title" : c3_data,
            "author" : c4_data
            }
            count=count+1
            print(count)
            data = json.dumps(temp_dic, indent=4)
            result_list.append(temp_dic)
            print(temp_dic)
    # except:
    #     json.dump(result_list, f, sort_keys = True, indent=4)
    #     f.close()
    #json.dumps(result_list, f, indent=4)

    json.dump(result_list, f, sort_keys = True, indent=4)
    f.close()
    print('Final count: ',count)
    print('Connection is closed!')
    return None

if __name__=="__main__":
    main()
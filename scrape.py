import urllib.parse
import urllib.request
from urllib.error import HTTPError
import json
import time
import logging
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
logger = logging.getLogger(__name__)

def tag_to_text(tag):
    if tag is not None:
        return tag.text
    else:
        return None

def drink_soup(soup):
    records = []
    for record in soup.listrecords.findAll('record'):
        records.append(
            {
                'id': tag_to_text(record.id), 
                'created': tag_to_text(record.created),
                'updated': tag_to_text(record.updated),
                'authors': [
                    {
                        'keyname': tag_to_text(author.keyname),
                        'forenames': [tag_to_text(forename) for forename in author.findAll('forenames')]
                    } 
                    for author in record.authors.findAll('author')
                ],
                'title': tag_to_text(record.title),
                'catagories': tag_to_text(record.categories).split(),
                'comments': tag_to_text(record.comments),
                'journal-ref': tag_to_text(record.find('journal-ref')),
                'license': tag_to_text(record.license),
                'abstract': tag_to_text(record.abstract)
            }
        )
    return records

cursor = None
finished = False
attempts = 1
all_records_count = 0

url = 'http://export.arxiv.org/oai2'
values = {
    'verb': 'ListRecords',
    'set': 'cs',
    'from': '1995-01-01',
    'until': '2018-01-01',
    'metadataPrefix': 'arXiv',
}

while not finished:

    data = urllib.parse.urlencode(values)
    data = data.encode('ascii')
    req = urllib.request.Request(url, data)
    
    
    try:
        with urllib.request.urlopen(req) as response:
            content = response.read()
            status = response.status
    except HTTPError:
        logging.info('HTTP error, attempt {} of 10, sleeping for 10 seconds'.format(attempts))
        attempts += 1
        if attempts >= 10:
            break
        time.sleep(10)
        continue
        
    attempts = 1    
    soup = BeautifulSoup(content, "lxml")
    records = drink_soup(soup)
    all_records_count += len(records)
    for record in records:
        print(json.dumps(record))
    resumptionToken = soup.find('resumptiontoken')
    cursor = resumptionToken.get('cursor')
    completeListSize = int(resumptionToken.get('completelistsize'))

    if not len(resumptionToken.text):
        logging.info('no resumption token, finishing')
        finished = True
    
    logging.info('downloaded {} of {} records'.format(all_records_count, completeListSize))
    values = {'verb': 'ListRecords', 'resumptionToken': resumptionToken.text}
    


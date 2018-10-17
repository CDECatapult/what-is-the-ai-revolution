import json
import requests
import logging
from os import makedirs
from os.path import join, exists
from datetime import date, timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
logger = logging.getLogger(__name__)

MY_API_KEY = open("creds_guardian.txt").read().strip()
API_ENDPOINT = 'http://content.guardianapis.com/search'
my_params = {
    'from-date': "",
    'to-date': "",
    'order-by': "newest",
    'show-fields': 'all',
    'page-size': 200,
    'api-key': MY_API_KEY,
    'q': 'technology'
}

# day iteration from here:
# http://stackoverflow.com/questions/7274267/print-all-day-dates-between-two-dates
start_date = date(2010, 1, 1)
end_date = date(2018,1, 1)
dayrange = range((end_date - start_date).days + 1)
for daycount in dayrange:
    dt = start_date + timedelta(days=daycount)
    datestr = dt.strftime('%Y-%m-%d')
    logging.info("Downloading {}".format(datestr))
    my_params['from-date'] = datestr
    my_params['to-date'] = datestr
    current_page = 1
    total_pages = 1
    while current_page <= total_pages:
        logging.info("...page {}".format(current_page))
        my_params['page'] = current_page
        resp = requests.get(API_ENDPOINT, my_params)
        data = resp.json()
        all_results = data['response']['results']
        for result in all_results:
            print(json.dumps(result))
        # if there is more than one page
        current_page += 1
        total_pages = data['response']['pages']



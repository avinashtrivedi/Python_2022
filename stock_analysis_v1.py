from lxml import html
import requests
import json
import argparse
import time
from fake_useragent import UserAgent
ua = UserAgent()
import numpy as np
from collections import OrderedDict
import random
import warnings
warnings.filterwarnings('ignore')

h_agent = ['Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US) AppleWebKit/525.19 (KHTML, like Gecko) Chrome/1.0.154.36 Safari/525.19',
'Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US) AppleWebKit/534.10 (KHTML, like Gecko) Chrome/7.0.540.0 Safari/534.10',
'Mozilla/5.0 (Windows; U; Windows NT 5.2; en-US) AppleWebKit/534.4 (KHTML, like Gecko) Chrome/6.0.481.0 Safari/534.4',
'Mozilla/5.0 (Macintosh; U; Intel Mac OS X; en-US) AppleWebKit/533.4 (KHTML, like Gecko) Chrome/5.0.375.86 Safari/533.4',
'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US) AppleWebKit/532.2 (KHTML, like Gecko) Chrome/4.0.223.3 Safari/532.2',
'Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US) AppleWebKit/532.0 (KHTML, like Gecko) Chrome/4.0.201.1 Safari/532.0',
'Mozilla/5.0 (Windows; U; Windows NT 5.2; en-US) AppleWebKit/532.0 (KHTML, like Gecko) Chrome/3.0.195.27 Safari/532.0',
'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US) AppleWebKit/530.5 (KHTML, like Gecko) Chrome/2.0.173.1 Safari/530.5',
'Mozilla/5.0 (Windows; U; Windows NT 5.2; en-US) AppleWebKit/534.10 (KHTML, like Gecko) Chrome/8.0.558.0 Safari/534.10',
'Mozilla/5.0 (X11; U; Linux x86_64; en-US) AppleWebKit/540.0 (KHTML,like Gecko) Chrome/9.1.0.0 Safari/540.0',
'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:20.0) Gecko/20100101 Firefox/20.0']


def parse(ticker):
    
    rand = random.randint(0,len(h_agent)-1)
    delays = [2, 1, 3]
    delay = np.random.choice(delays)
   
    headers={'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3',
        'Accept-Language': 'en-US,en;q=0.9,es;q=0.8',
        'Host': 'stockanalysis.com',
         'Accept-Encoding': 'gzip',
        'Upgrade-Insecure-Requests': '1',
        'Referer':'https://www.google.com/',
        'User-Agent': ua.chrome,
        'X-Requested-With': 'XMLHttpRequest'
        }
    
    
    
#     print('Hreader:->',rand+1)
    url = "https://stockanalysis.com/stocks/{}/financials/cash-flow-statement".format(ticker)
#     headers={'User-Agent': h_agent[rand]}
    response = requests.get(url, headers=headers)
    parser = html.fromstring(response.content)
    fcfs = parser.xpath('//table[contains(@id,"financial-table")]//tr[td/span/text()[contains(., "Free Cash Flow")]]')[0].xpath('.//td/span/text()')[1:]
    last_fcf = float(fcfs[0].replace(',', ''))
    time.sleep(delay)
    url = "https://finance.yahoo.com/quote/{}/analysis?p={}".format(ticker, ticker)
    response = requests.get(url, headers= {'User-Agent': ua.chrome})
    parser = html.fromstring(response.content)
    ge = parser.xpath('//table//tbody//tr')

    for row in ge:
        label = row.xpath("td/span/text()")[0]
        if 'Next 5 Years' in label:
            try:
                ge = float(row.xpath("td/text()")[0].replace('%', ''))
            except:
                ge = []
            break

    time.sleep(delay)
    url = "https://stockanalysis.com/stocks/{}/".format(ticker)
    response = requests.get(url, headers=headers)
    parser = html.fromstring(response.content)
    shares = parser.xpath('//div[@class="order-1 flex flex-row gap-4"]//table//tbody//tr[td/text()[contains(., "Shares Out")]]')

    shares = shares[0].xpath('td/text()')[1]
    factor = 1000 if 'B' in shares else 1 
    shares = float(shares.replace('B', '').replace('M', '')) * factor

    time.sleep(delay)
    url = "https://stockanalysis.com/stocks/{}/financials/".format(ticker)
    response = requests.get(url, headers=headers)
    parser = html.fromstring(response.content)
    eps = parser.xpath('//table[contains(@id,"financial-table")]//tr[td/span/text()[contains(., "EPS (Diluted)")]]')[0].xpath('.//td/span/text()')[1:]
    eps = float(eps[0].replace(",", ""))
    market_price = float(parser.xpath('//div[@class="p"]/text()')[0].replace('$', '').replace(',', ''))
    return {'fcf': last_fcf, 'ge': ge, 'yr': 5, 'dr': 10, 'pr': 2.5, 'shares': shares, 'eps': eps, 'mp': market_price}

def dcf(data):
    forecast = [data['fcf']]

    if data['ge'] == []:
    	raise ValueError("No growth rate available from Yahoo Finance")

    for i in range(1, data['yr']):
        forecast.append(round(forecast[-1] + (data['ge'] / 100) * forecast[-1], 2))

    forecast.append(round(forecast[-1] * (1 + (data['pr'] / 100)) / (data['dr'] / 100 - data['pr'] / 100), 2)) #terminal value
    discount_factors = [1 / (1 + (data['dr'] / 100))**(i + 1) for i in range(len(forecast) - 1)]

    pvs = [round(f * d, 2) for f, d in zip(forecast[:-1], discount_factors)]
    pvs.append(round(discount_factors[-1] * forecast[-1], 2)) # discounted terminal value
    
    print("Forecasted cash flows: {}".format(", ".join(map(str, forecast))))
    print("PV of cash flows: {}".format(", ".join(map(str, pvs))))

    dcf = sum(pvs)
    print("Fair value: {}\n".format(dcf / data['shares']))

def reverse_dcf(data):
    pass

def graham(data):
    if data['eps'] > 0:
        expected_value = data['eps'] * (8.5 + 2 * (data['ge']))
        ge_priced_in = (data['mp'] / data['eps'] - 8.5) / 2

        print("Expected value based on growth rate: {}".format(expected_value))
        print("Growth rate priced in for next 7-10 years: {}\n".format(ge_priced_in))
    else:
        print("Not applicable since EPS is negative.")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('ticker', help='Ticker to analyse. Example: GOOG')
    argparser.add_argument('--discount_rate', help='Discount rate in %. Default: 10', default=10)
    argparser.add_argument('--growth_estimate', help='Estimated yoy growth rate. Default: Fetched from Yahoo Finance')
    argparser.add_argument('--terminal_rate', help='Terminal growth rate. Default: 2.5')
    argparser.add_argument('--period', help='Time period in years. Default: 5')
    args = argparser.parse_args()
    
    ticker = args.ticker
#     ticker = input('Enter Stock: ')
    print("Fetching data for %s...\n" % (ticker))
    data = parse(ticker)
    print("=" * 80)
    print("DCF model (basic)")
    print("=" * 80 + "\n")

    if args.period is not None:
        data['yr'] = int(args.period)
    if args.growth_estimate is not None:
        data['ge'] = float(args.growth_estimate)
    if args.discount_rate is not None:
        data['dr'] = float(args.discount_rate)
    if args.terminal_rate is not None:
        data['pr'] = float(args.terminal_rate)

    print("Market price: {}".format(data['mp']))
    print("EPS: {}".format(data['eps']))
    print("Growth estimate: {}".format(data['ge']))
    print("Term: {} years".format(data['yr']))
    print("Discount Rate: {}%".format(data['dr']))
    print("Perpetual Rate: {}%\n".format(data['pr']))

    dcf(data)

    print("=" * 80)
    print("Graham style valuation basic (Page 295, The Intelligent Investor)")
    print("=" * 80 + "\n")

    graham(data)
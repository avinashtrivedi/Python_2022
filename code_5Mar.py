from lxml import html
import requests
import json
import argparse
from collections import OrderedDict

import warnings
warnings.filterwarnings('ignore')

def parse(ticker):
    url = "https://stockanalysis.com/stocks/{}/financials/cash-flow-statement".format(ticker)
    response = requests.get(url, verify=False)
    parser = html.fromstring(response.content)
    fcfs = parser.xpath('//table[contains(@id,"financial-table")]//tr[td/span/text()[contains(., "Free Cash Flow")]]')[0].xpath('.//td/span/text()')[1:]
    last_fcf = float(fcfs[0].replace(',', ''))
    
    url = "https://finance.yahoo.com/quote/{}/analysis?p={}".format(ticker, ticker)
    response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:20.0) Gecko/20100101 Firefox/20.0'})
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

    url = "https://stockanalysis.com/stocks/{}/".format(ticker)
    response = requests.get(url, verify=False)
    parser = html.fromstring(response.content)
    shares = parser.xpath('//div[@class="order-1 flex flex-row gap-4"]//table//tbody//tr[td/text()[contains(., "Shares Out")]]')

    shares = shares[0].xpath('td/text()')[1]
    factor = 1000 if 'B' in shares else 1 
    shares = float(shares.replace('B', '').replace('M', '')) * factor

    url = "https://stockanalysis.com/stocks/{}/financials/".format(ticker)
    response = requests.get(url, verify=False)
    parser = html.fromstring(response.content)
    eps = parser.xpath('//table[contains(@id,"financial-table")]//tr[td/span/text()[contains(., "EPS (Diluted)")]]')[0].xpath('.//td/span/text()')[1:]
    eps = float(eps[0].replace(",", ""))
    market_price = float(parser.xpath('//div[@class="price-ext"]/text()')[0].replace('$', '').replace(',', ''))
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
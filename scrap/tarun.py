#!/usr/bin/env python
# coding: utf-8

# In[ ]:
def shopee_scraper():
    import sys
    import requests
    import numpy as np
    from requests.exceptions import ConnectionError, ReadTimeout
    import warnings
    import pandas as pd
    from time import sleep
    from tqdm.notebook import tqdm
    from json.decoder import JSONDecodeError
    warnings.filterwarnings("ignore")
    headers = {'User-Agent': 'Chrome'}
    headers1 = {'User-Agent': 'Chrome' ,'if-none-match': '2b20b0c3491db9ea8eec4f498fed3cc3',
'if-none-match-': '55b03-71a3d43a62d13062a6e94e90f793bda'}
    countries = ['indonesia','taiwan','vietnam','thailand','philippines','malaysia','singapore','brazil']
    c1 = ['indonesia','taiwan','vietnam','thailand','philippines']
    c2 = ['malaysia','singapore','brazil']

    print('This Code is created by Tarun Kumar. Feel free to reach in case of any problem.')
    print('-'*60)

    country = input('Enter the name of country.  ').lower()
    print('-'*60)

    while country not in countries:
        country = input('You have entered wrong country. Please enter the correct spelling.  ').lower()
        print('-'*60) 

    keyword = input('Enter the keyword.  ').lower()
    print('-'*60)

    page_numbers = str(input('Enter the number of pages.  '))
    print('-'*60) 

    output_folder = input('Enter the path of folder where you want output file.  ')
    print('-'*60)

    ofile = input('Enter the name of output file.  ')
    print('-'*60)

    if country == 'indonesia':
        domain = 'https://shopee.co.id/'
        idomain = 'https://cf.shopee.co.id/file/'
        curr = 'IDR'
    elif country == 'taiwan':
        domain = 'https://shopee.tw/'
        idomain = 'https://cf.shopee.tw/file/'
        curr = 'TWD'
    elif country == 'vietnam':
        domain = 'https://shopee.vn/'
        idomain = 'https://cf.shopee.vn/file/'
        curr = 'VND'
    elif country == 'thailand':
        domain = 'https://shopee.co.th/'
        idomain = 'https://cf.shopee.co.th/file/'
        curr = 'THB'
    elif country == 'philippines':
        domain = 'https://shopee.ph/'
        idomain = 'https://cf.shopee.ph/file/'
        curr = 'PHP'
    elif country == 'malaysia':
        domain = 'https://shopee.com.my/'
        idomain = 'https://cf.shopee.com.my/file/'
        curr = 'MYR'
    elif country == 'singapore':
        domain = 'https://shopee.sg/'
        idomain = 'https://cf.shopee.sg/file/'
        curr = 'SGD'
    elif country == 'brazil':
        domain = 'https://shopee.com.br/'
        idomain = 'https://cf.shopee.com.br/file/'
        curr = 'BRL'

    def connected_or_not():
        try:
            requests.get('http://google.com')
            return 'Connected'
        except:
            return 'Not Connected'
        
    def remove_char(text):
        text = str(text).replace("'","").replace("[","").replace("]","")
        return text

    titles_list = []
    item_number = []
    shop_number = []
    sold = []
    stock = []
    price = []
    Url = []
    description = []
    img_url = []
    seller_loc = []
    prod_url = []
    seller_url = []
    seller_id = []
    store_id = []
    rating = []
    store_url = []
    item_id = []
    req_len = []
    final_price = []
    length=[]
    new = 0

    print('Starting getting listings') 

    with tqdm(total=len(range(int(page_numbers))), desc = 'Listing') as pbar:
        for i in range(int(page_numbers)):
            base_url = domain+'api/v2/search_items/?by=relevancy&keyword='+keyword+'&limit=50&newest='+str(new)+'&order=desc&page_type=search&version=2'
            try:
                r = requests.get(base_url, headers = headers1, timeout = 20).json()
            except (ConnectionError, ReadTimeout, JSONDecodeError):
                print('Waiting for 15 seconds so connection could re-establish.')
                sleep(15)
                try:
                    r = requests.get(base_url, headers = headers1, timeout = 20).json()
                except (ConnectionError, ReadTimeout, JSONDecodeError):
                    if connected_or_not() == 'Connected':
                        sys.exit('YOU ARE BLOKED BY SHOPEE. TRY AFTER SOME TIME.')
                    elif connected_or_not() == 'Not Connected':
                        while connected_or_not() == 'Not Connected':
                            connected_or_not()
                            print('No Internet', end = '\r')
                            sleep(15)
                        print('Internet is back. Listing again.')
                        r = requests.get(base_url, headers = headers1, timeout = 20).json()
            for item in r['items']:
                item_number.append(item['itemid'])
                shop_number.append(item['shopid'])
            pbar.update(1)
            new = new + 50
            sleep(np.random.randint(1,5))

    print('Listings are done.')
    print('-'*60) 

    print('Starting profiling.')                  
    for i in range(len(item_number)):
        prod_url.append(domain + '/api/v2/item/get?itemid=' + str(item_number[i]) +'&shopid=' + str(shop_number[i]))
        seller_url.append(domain + 'api/v2/shop/get?shopid=' + str(shop_number[i]))

    with tqdm(total=len(prod_url), desc = 'Profiling') as pbar1:
        for i in range(len(prod_url)):
            try:
                r1 = requests.get(prod_url[i], headers = headers, timeout = 20).json()
            except (ConnectionError, ReadTimeout, JSONDecodeError):
                print('Waiting for 15 seconds so connection could re-establish.')
                sleep(15)
                try:
                    r1 = requests.get(prod_url[i], headers = headers, timeout = 20).json()
                except (ConnectionError, ReadTimeout, JSONDecodeError):
                    if connected_or_not() == 'Connected':
                        sys.exit('YOU ARE BLOKED BY SHOPEE. TRY AFTER SOME TIME.')
                    elif connected_or_not() == 'Not Connected':
                        while connected_or_not() == 'Not Connected':
                            connected_or_not()
                            print('No Internet', end = '\r')
                            sleep(15)
                        print('Internet is back. Profiling again.')
                        r1 = requests.get(prod_url[i], headers = headers, timeout = 20).json()
            if r1['item'] != None:
                titles_list.append(r1['item']['name'])
                sold.append(r1['item']['historical_sold'])
                seller_loc.append(r1['item']['shop_location'])
                description.append(r1['item']['description'])
                stock.append(r1['item']['stock'])
                price.append(str(r1['item']['price_min']))
                img_url.append(r1['item']['images'])
            elif r1['item'] == None :
                titles_list.append('NA')
                sold.append('NA')
                seller_loc.append('NA')
                description.append('NA')
                stock.append('NA')
                price.append('NA')
                img_url.append(['NA'])          
            try:
                r2 = requests.get(seller_url[i], headers = headers, timeout = 20).json()
            except (ConnectionError, ReadTimeout, JSONDecodeError):
                print('Waiting for 15 seconds so connection could re-establish.')
                sleep(15)
                try:
                    r2 = requests.get(seller_url[i], headers = headers, timeout = 20).json()
                except (ConnectionError, ReadTimeout, JSONDecodeError):
                    if connected_or_not() == 'Connected':
                        sys.exit('YOU ARE BLOKED BY SHOPEE. TRY AFTER SOME TIME.')
                    elif connected_or_not() == 'Not Connected':
                        while connected_or_not() == 'Not Connected':
                            connected_or_not()
                            print('No Internet', end = '\r')
                            sleep(15)
                        print('Internet is back. Profiling again.')
                        r2 = requests.get(seller_url[i], headers = headers, timeout = 20).json()
            if r2['data'] != None:
                store_url.append(domain + r2['data']['account']['username'])
                seller_id.append(r2['data']['account']['username'])
                store_id.append(r2['data']['name'])
                star = r2['data']['rating_star']
                if star != None:
                    rating.append(round(star,1))
                elif star == None:
                    rating.append(0)
                else:
                    rating.append(star)
            elif r2['data'] == None:
                store_url.append('NA')
                seller_id.append('NA')
                store_id.append('NA')
                rating.append('NA')
            pbar1.update(1)

    for i in range(len(titles_list)):
        Url.append(domain + titles_list[i].replace(' ', '-') + '-i.'+ str(shop_number[i])+ '.'+ str(item_number[i]))

    for i in range(len(Url)):
        Url[i] = Url[i].replace('%',"")
        Url[i] = Url[i].replace('#',"")
        Url[i] = Url[i].replace('?',"")
    
    for img in img_url:
        for i in range(len(img)):
            img[i] = idomain + img[i]

    for i in range(len(item_number)):
        item_id.append('i.' + str(shop_number[i]) + '.' + str(item_number[i]))

    for i in price:
        length.append(len(i))

    if country in c1:
        for g in length:
            if g == 10:
                req_len.append(5)
            elif g < 10:
                req_len.append(int((g - (10-g))/2))
            elif g > 10:
                req_len.append(int((g + (g-10))/2))    

        for i in range(len(price)):
            final_price.append(curr + price[i][:req_len[i]])

    elif country in c2:  
        for g in length:
            if g == 6:
                req_len.append(3)
            elif g < 6:
                req_len.append(int((g - (6-g))/2))
            elif g > 6:
                req_len.append(int((g + (g-6))/2))    

        for i in range(len(price)):
            final_price.append(price[i][:req_len[i]])

        for i in range(len(final_price)):
            if len(final_price[i]) == 2:
                final_price[i] = curr + "." + final_price[i]
            elif len(final_price[i]) > 2:
                first = final_price[i][:-2]
                last = final_price[i][-2:]
                final_price[i] = curr +first+"."+ last

    data = pd.DataFrame({'URL':Url, 'Title':titles_list, 'Item Number':item_id, 'Price':final_price, 'Sold':sold, 'Stock':stock, 'Sellername': seller_id, 'Under store name': store_id, 'Store Url': store_url, 'Rating':rating, 'Description':description, 'Seller Location':seller_loc, 'Image Url': img_url})
    data.drop(data[data['Title'] == 'NA'].index, inplace = True)
    data['Image Url'] = data['Image Url'].apply(remove_char)
    data.to_excel(output_folder + '\\' + ofile + '.xlsx')
    data.to_csv(output_folder + '\\' + ofile + '.csv',encoding="utf-8-sig")
    print('Demo of the output file')
    return data.head()



def shopee_screenshot():
    import pandas as pd
    from time import sleep
    from tqdm.notebook import tqdm
    from selenium import webdriver
    from selenium.webdriver.common.keys import Keys
    from selenium.common.exceptions import TimeoutException, WebDriverException
    options = webdriver.ChromeOptions()
    options.headless = True
    print('This Code is created by Tarun Kumar. Feel free to reach in case of any problem. ')
    print('-'*60)
    input_folder = input('Enter the path of folder where excel file is stored. ')
    print('-'*60)
    file = input('Enter file name with .xlsx & file must have a header example(url/URL). ')
    print('-'*60)
    output_folder = input('Enter the path of folder where you want screenshots. ')
    print('-'*60)
    print('Starting capturing screenshots.')
    data = pd.read_excel(input_folder +"\\" + file) 
    with tqdm(total=len(data), desc = 'Capturing') as pbar:
        with webdriver.Chrome(r"C:\chromedriver_win32\chromedriver.exe",options=options) as driver:
            for link in data.iloc[:,0]:
                str1 = link.split('-')[-1]
                image = str1.replace('.','_') + '.png'
                try:
                    driver.get(link)
                    sleep(1.5)
                except TimeoutException:
                    print('Waiting for 2 minutes so connection could re-establish.')
                    sleep(120)
                    driver.get(link)
                    sleep(1.5)
                for i in range(4): 
                    driver.find_element_by_tag_name('body').send_keys(Keys.PAGE_DOWN)
                    sleep(1)
                driver.find_element_by_tag_name('body').send_keys(Keys.HOME)
                S = lambda X: driver.execute_script('return document.body.parentNode.scroll'+X)
                driver.set_window_size(S('Width'),S('Height')) 
                try:
                    driver.find_element_by_tag_name('body').screenshot(output_folder+"\\"+image)
                except WebDriverException:
                    print(link + ' is not saved')
                pbar.update(1)
    print('All Screenshots are done. Check your folder')
    
def shopee_status_checker():
    import sys
    import requests
    import warnings
    warnings.filterwarnings("ignore")
    import pandas as pd
    from time import sleep
    headers = {'User-Agent': 'Chrome'}
    from tqdm.notebook import tqdm
    from requests.exceptions import ConnectionError, ReadTimeout
    from json.decoder import JSONDecodeError
    print('This Code is created by Abhishek Chauhan. Feel free to reach in case of any problem.')
    print('-'*60)
    input_folder = input('Enter the path of folder where excel file is stored.')
    print('-'*60)
    ifile = input('Enter file name with .xlsx & file must have a header example(url/URL)')
    print('-'*60)
    output_folder = input('Enter the path of folder where you want results.')
    print('-'*60)
    ofile = input('Enter the name of output file')
    print('-'*60)
    print('Starting checking status')
    def connected_or_not():
        try:
            requests.get('http://google.com')
            return 'Connected'
        except:
            return 'Not Connected'
    dt = pd.read_excel(input_folder +"\\" + ifile)
    St = []
    with tqdm(total=len(dt), desc = 'Progress') as pbar:
        for l in dt.iloc[:,0]:
            s = l.split('.')[-2]
            p = l.split('.')[-1]
            dom = l.split('/')[0] + '//' + l.split('/')[2]
            bu = dom + '/api/v2/item/get?itemid=' + p +'&shopid=' + s
            try:
                d = requests.get(bu, headers = headers, timeout = 20).json()
            except (ConnectionError, ReadTimeout, JSONDecodeError):
                print('Waiting for 20 seconds so connection could re-establish.')
                sleep(20)
                try:
                    d = requests.get(bu, headers = headers, timeout = 20).json()
                except (ConnectionError, ReadTimeout, JSONDecodeError):
                    if connected_or_not() == 'Connected':
                        sys.exit('YOU ARE BLOKED BY SHOPEE. TRY AFTER SOME TIME.')
                    elif connected_or_not() == 'Not Connected':
                        while connected_or_not() == 'Not Connected':
                            connected_or_not()
                            print('No Internet', end = '\r')
                            sleep(15)
                        print('Internet is back. Checking status again.')
                        d = requests.get(bu, headers = headers, timeout = 20).json()
            if d['item'] == None:
                St.append('Removed')            
            elif d['item']['item_status'] == 'banned':
                St.append('Removed') 
            elif d['item']['item_status'] == 'normal':
                St.append('Active')
            elif d['item']['item_status'] == 'sold_out':
                St.append('Active')
            else:
                St.append(d['item']['item_status'])
            pbar.update(1)
    st = pd.DataFrame({'Status':St})
    fn = pd.concat([dt,st],axis=1)
    print('Demo of the output file')
    fn.to_csv(output_folder +"\\" + ofile + '.csv',encoding="utf-8-sig")
    return fn.head(10)

def shopee_profiler():
    import sys
    import requests
    import warnings
    import pandas as pd
    from time import sleep
    warnings.filterwarnings("ignore")
    headers = {'User-Agent': 'Chrome'}
    from tqdm.notebook import tqdm
    from requests.exceptions import ConnectionError, ReadTimeout
    from json.decoder import JSONDecodeError
    c1 = ['indonesia','taiwan','vietnam','thailand','philippines']
    c2 = ['malaysia','singapore','brazil']

    print('This Code is created by tarun kumar. Feel free to reach in case of any problem.')
    print('-'*60)

    input_folder = input('Enter the path of folder where excel file is stored.')
    print('-'*60)

    ifile = input('Enter file name with .xlsx & file must have a header example(url/URL)')
    print('-'*60)

    output_folder = input('Enter the path of folder where you want output file.  ')
    print('-'*60)

    ofile = input('Enter the name of output file.  ')
    print('-'*60)

    def connected_or_not():
        try:
            requests.get('http://google.com')
            return 'Connected'
        except:
            return 'Not Connected'
        
    def remove_char(text):
        text = str(text).replace("'","").replace("[","").replace("]","")
        return text

    titles_list = []
    sold = []
    stock = []
    price = []
    description = []
    img_url = []
    seller_loc = []
    seller_id = []
    store_id = []
    rating = []
    store_url = []
    final_price = []

    dt = pd.read_excel(input_folder +"\\" + ifile)

    with tqdm(total=len(dt), desc = 'Profiling') as pbar:
        for l in dt.iloc[:,0]:
            s = l.split('.')[-2]
            p = l.split('.')[-1]
            dom = l.split('/')[0] + '//' + l.split('/')[2]
            if dom == 'https://shopee.co.id':
                country = 'indonesia'
                idomain = 'https://cf.shopee.co.id/file/'
                curr = 'IDR' 
            elif dom == 'https://shopee.tw':
                country = 'taiwan'
                idomain = 'https://cf.shopee.tw/file/'
                curr = 'TWD'
            elif dom == 'https://shopee.vn':
                country = 'vietnam'
                idomain = 'https://cf.shopee.vn/file/'
                curr = 'VND'
            elif dom == 'https://shopee.co.th':
                country = 'thailand'
                idomain = 'https://cf.shopee.co.th/file/'
                curr = 'THB'
            elif dom == 'https://shopee.ph':
                country = 'philippines'
                idomain = 'https://cf.shopee.ph/file/'
                curr = 'PHP'
            elif dom == 'https://shopee.com.my':
                country = 'malaysia'
                idomain = 'https://cf.shopee.com.my/file/'
                curr = 'MYR'
            elif dom == 'https://shopee.sg':
                country = 'singapore'
                idomain = 'https://cf.shopee.sg/file/'
                curr = 'SGD'
            elif dom == 'https://shopee.com.br':
                country = 'brazil'
                idomain = 'https://cf.shopee.com.br/file/'
                curr = 'BRL'
            bu = dom + '/api/v2/item/get?itemid=' + p +'&shopid=' + s
            su = dom + '/api/v2/shop/get?shopid=' + s
            try:
                r1 = requests.get(bu, headers = headers, timeout = 20).json()
            except (ConnectionError, ReadTimeout, JSONDecodeError):
                print('Waiting for 20 seconds so connection could re-establish.')
                sleep(20)
                try:
                    r1 = requests.get(bu, headers = headers, timeout = 20).json()
                except (ConnectionError, ReadTimeout, JSONDecodeError):
                    if connected_or_not() == 'Connected':
                        sys.exit('YOU ARE BLOKED BY SHOPEE. TRY AFTER SOME TIME.')
                    elif connected_or_not() == 'Not Connected':
                        while connected_or_not() == 'Not Connected':
                            connected_or_not()
                            print('No Internet', end = '\r')
                            sleep(15)
                        print('Internet is back. Checking status again.')
                        r1 = requests.get(bu, headers = headers, timeout = 20).json()
            if r1['item'] != None:
                titles_list.append(r1['item']['name'])
                sold.append(r1['item']['historical_sold'])
                seller_loc.append(r1['item']['shop_location'])
                description.append(r1['item']['description'])
                stock.append(r1['item']['stock'])
                img1 = r1['item']['images']
                for i in range(len(img1)):
                    img1[i] = idomain + img1[i]
                img_url.append(img1)
                p = str(r1['item']['price_min'])
                l = len(p)
                if country in c1:
                    if l == 10:
                        req_len = 5
                    elif l < 10:
                        req_len = int((l - (10-l))/2)
                    elif l > 10:
                        req_len = int((l + (l-10))/2)
                    final_price.append(curr + p[:req_len])

                elif country in c2:  
                    if l == 6:
                        req_len = 3
                    elif l < 6:
                        req_len = int((l - (6-l))/2)
                    elif l > 6:
                        req_len = int((l + (l-6))/2) 

                    if len(p[:req_len]) == 2:
                        final_price.append(curr + "." + p[:req_len])
                    elif len(p[:req_len]) > 2:
                        first = p[:req_len][:-2]
                        last = p[:req_len][-2:]

                        final_price.append(curr +first+"."+ last)          

            elif r1['item'] == None :
                titles_list.append('NA')
                sold.append('NA')
                seller_loc.append('NA')
                description.append('NA')
                stock.append('NA')
                final_price.append('NA')
                img_url.append('NA')          
            try:
                r2 = requests.get(su, headers = headers, timeout = 20).json()
            except (ConnectionError, ReadTimeout, JSONDecodeError):
                print('Waiting for 20 seconds so connection could re-establish.')
                sleep(20)
                try:
                    r2 = requests.get(su, headers = headers, timeout = 20).json()
                except (ConnectionError, ReadTimeout, JSONDecodeError):
                    if connected_or_not() == 'Connected':
                        sys.exit('YOU ARE BLOKED BY SHOPEE. TRY AFTER SOME TIME.')
                    elif connected_or_not() == 'Not Connected':
                        while connected_or_not() == 'Not Connected':
                            connected_or_not()
                            print('No Internet', end = '\r')
                            sleep(15)
                        print('Internet is back. Profiling again.')
                        r2 = requests.get(su, headers = headers, timeout = 20).json()
            if r2['data'] != None:
                store_url.append(dom + '/' + r2['data']['account']['username'])
                seller_id.append(r2['data']['account']['username'])
                store_id.append(r2['data']['name'])
                star = r2['data']['rating_star']
                if star != None:
                    rating.append(round(star,1))
                elif star == None:
                    rating.append(0)
                else:
                    rating.append(star)
            elif r2['data'] == None:
                store_url.append('NA')
                seller_id.append('NA')
                store_id.append('NA')
                rating.append('NA')
            pbar.update(1)

    data = pd.DataFrame({'Title':titles_list, 'Price':final_price,  'Sold':sold,  'Stock':stock, 'Seller name': seller_id, 'Under store name':store_id, 'Store Url': store_url, 'Rating':rating, 'Description':description, 'Seller Location':seller_loc, 'Image Url': img_url})
    fn = pd.concat([dt,data],axis=1)
    fn['Image Url'] = fn['Image Url'].apply(remove_char)
    fn.to_excel(output_folder + '\\' + ofile + '.xlsx')
    fn.to_csv(output_folder + '\\' + ofile + '.csv',encoding="utf-8-sig")
    print('Demo of the output file')
    return fn.head()

def tokopedia_scraper():
    import sys
    import re
    import json
    import numpy as np
    import requests
    import pandas as pd
    from time import sleep
    from bs4 import BeautifulSoup
    from selenium import webdriver
    from tqdm.notebook import tqdm
    from selenium.webdriver.common.keys import Keys
    from requests.exceptions import ConnectionError, ReadTimeout
    from selenium.common.exceptions import NoSuchElementException
    options = webdriver.ChromeOptions()
    options.headless = True
    options.add_argument("--disable-notifications")
    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.61 Safari/537.36'
    options.add_argument(f'user-agent={user_agent}')
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.61 Safari/537.36'}

    print('This Code is created by Abhishek Chauhan. Feel free to reach in case of any problem.')
    print('-'*60)

    input_link = input("Enter the search page link  ")
    print('-'*60)

    output_folder = input('Enter the path of folder where you want output file.  ')
    print('-'*60)

    ofile = input('Enter the name of output file.  ')
    print('-'*60)

    def connected_or_not():
        try:
            requests.get('http://google.com')
            return 'Connected'
        except:
            return 'Not Connected'

        
    def remove_char(text):
        text = str(text).replace("'","").replace("[","").replace("]","")
        return text
    
    link = []
    links = []
    price = []
    title = []
    loc = []
    seller = []
    store = []
    sold = []
    stock = []
    desc = []
    img = []
    img_url = []
    item =[]

    print('Starting Listings')
    print('-'*60)

    with webdriver.Chrome(r"C:\chromedriver_win32\chromedriver.exe",options=options) as driver:
        driver.get(input_link)
        sleep(3)
        for i in range(5): 
            driver.find_element_by_tag_name('body').send_keys(Keys.PAGE_DOWN)
            sleep(0.5)
        with tqdm(desc = 'Listing') as pbar:
            while True:
                soup1 = BeautifulSoup(driver.page_source, features="lxml")
                for i in soup1.findAll('div', class_ = "css-1g20a2m"):
                    link.append(i.a.get('href'))           
                try:
                    html = driver.find_element_by_tag_name('html')
                    html.send_keys(Keys.END)
                    sleep(2)
                    element = driver.find_element_by_class_name('css-1gpfbae-unf-pagination-item')
                    driver.execute_script("arguments[0].click();", element)
                    #elm = driver.find_element_by_class_name('css-1gpfbae-unf-pagination-item')
                    #elm.click()
                    #element = driver.find_element_by_css('div[class*="loadingWhiteBox"]')
                    #webdriver.ActionChains(driver).move_to_element(element ).click(element ).perform()
                    sleep(3)
                    for i in range(5): 
                        driver.find_element_by_tag_name('body').send_keys(Keys.PAGE_DOWN)
                        sleep(0.5)
                except NoSuchElementException:
                    print('No more pages left')
                    break
                pbar.update(1)

    Link = []
    for i in range(len(link)):
        if 'ta.tokopedia.com' not in link[i]:
            Link.append(link[i])

    print('Listings are done.')
    print('-'*60) 

    print('Starting profiling.')
    print('-'*60) 

    with tqdm(total=len(Link), desc = 'Profiling') as pbar1:
        for i in range(len(Link)):
            try:
                r1 = requests.get(Link[i], headers = headers, timeout = 20)
            except (ConnectionError, ReadTimeout):
                print('Waiting for 2 minutes so connection could re-establish.')
                sleep(120)
                try:
                    r1 = requests.get(Link[i], headers = headers, timeout = 20)
                except (ConnectionError, ReadTimeout):
                    if connected_or_not() == 'Connected':
                        sys.exit('YOU ARE BLOKED BY TOKOPEIDA. TRY AFTER SOME TIME.')
                    elif connected_or_not() == 'Not Connected':
                        while connected_or_not() == 'Not Connected':
                            connected_or_not()
                            print('No Internet', end = '\r')
                            sleep(15)
                        print('Internet is back. Profiling again.')
                        r1 = requests.get(Link[i], headers = headers, timeout = 20)
            soup = BeautifulSoup(r1.content, features="lxml")
            links.append(r1.url.split("?")[0])
            item.append(r1.url.split("?")[0].split("/",3)[-1])
            store.append(r1.url.split("?")[0].rsplit("/", 1)[0])
            try:
                price.append(soup.find('h3', class_ = 'css-a94u6c').text)
            except AttributeError:
                price.append('NA')
            try:
                title.append(soup.find('h1', class_ ="css-x7lc0h").text)
            except AttributeError:
                title.append('NA')
            try:
                seller.append(soup.find('a', class_ = 'css-1cwp34r').text)
            except AttributeError:
                seller.append('NA')
            try:
                loc.append((soup.find('span',attrs={'data-testid' : 'lblPDPFooterLastOnline'}).text).split('\xa0')[0])
            except AttributeError:
                loc.append('NA')
            try:
                sell = re.search('\d+', soup.find('span',attrs={'data-testid' : 'lblPDPDetailProductSuccessRate'}).text).group(0)
                sold.append(int(sell))
            except AttributeError:
                sold.append(0)
            try:
                stok = re.search('\d+', soup.find('p',attrs={'data-testid' : 'lblPDPDetailProductStock'}).text).group(0)
                stock.append(int(stok))
            except AttributeError:
                stock.append(1)  
            try:
                json1 = json.loads(soup.findAll('script', type = "application/ld+json")[1].string, strict = False)
                desc.append(json1['description'])
            except IndexError:
                desc.append('NA')
            for i in soup.findAll('img', class_ = 'success'):
                img.append(i.get('src').replace('200-square', '700'))
            img_url.append(img)
            img = []
            pbar1.update(1)
            #sleep(np.random.randint(1,5))
    data = pd.DataFrame({'URL':links, 'Item number': item, 'Price':price, 'Title': title, 'Seller Location':loc, 'Store Url': store, 'Seller ID': seller, 'Sold':sold, 'Stock': stock, 'Description': desc, 'Img Url': img_url})
    data = data.drop_duplicates('Item number', keep='first')
    data.drop(data[data['Title'] == 'NA'].index, inplace = True)
    data['Img Url'] = data['Img Url'].apply(remove_char)
    data.to_excel( output_folder + '\\' + ofile + '.xlsx', index=False)
    data.to_csv( output_folder + '\\' + ofile + '.csv', index=False, encoding="utf-8-sig")
    print('Demo of the file. Duplicates are automatically removed')
    return data.head()

def tokopedia_profiler():
    import re
    import sys
    import json
    import requests
    import numpy as np
    import pandas as pd
    from time import sleep
    from bs4 import BeautifulSoup
    from tqdm.notebook import tqdm
    from requests.exceptions import ConnectionError, ReadTimeout
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.61 Safari/537.36'}

    print('This Code is created by Abhishek Chauhan. Feel free to reach in case of any problem.')
    print('-'*60)

    input_folder = input('Enter the path of folder where excel file is stored.')
    print('-'*60)

    ifile = input('Enter file name with .xlsx & file must have a header example(url/URL)')
    print('-'*60)

    output_folder = input('Enter the path of folder where you want output file.  ')
    print('-'*60)

    ofile = input('Enter the name of output file.  ')
    print('-'*60)

    def connected_or_not():
        try:
            requests.get('http://google.com')
            return 'Connected'
        except:
            return 'Not Connected'

    def remove_char(text):
            text = str(text).replace("'","").replace("[","").replace("]","")
            return text

    link = []
    links = []
    price = []
    title = []
    loc = []
    seller = []
    store = []
    sold = []
    stock = []
    desc = []
    img = []
    img_url = []
    item =[]

    dt = pd.read_excel(input_folder +"\\" + ifile)

    print('Starting profiling.')
    print('-'*60) 

    with tqdm(total=len(dt), desc = 'Profiling') as pbar1:
        for link in dt.iloc[:,0]:
            try:
                r1 = requests.get(link, headers = headers, timeout = 20)
            except (ConnectionError, ReadTimeout):
                print('Waiting for 2 minutes so connection could re-establish.')
                sleep(120)
                try:
                    r1 = requests.get(link, headers = headers, timeout = 20)
                except (ConnectionError, ReadTimeout):
                    if connected_or_not() == 'Connected':
                        sys.exit('YOU ARE BLOKED BY TOKOPEIDA. TRY AFTER SOME TIME.')
                    elif connected_or_not() == 'Not Connected':
                        while connected_or_not() == 'Not Connected':
                            connected_or_not()
                            print('No Internet', end = '\r')
                            sleep(15)
                        print('Internet is back. Profiling again.')
                        r1 = requests.get(link, headers = headers, timeout = 20)
            soup = BeautifulSoup(r1.content, features="lxml")
            links.append(r1.url.split("?")[0])
            item.append(r1.url.split("?")[0].split("/",3)[-1])
            store.append(r1.url.split("?")[0].rsplit("/", 1)[0])
            try:
                price.append(soup.find('div', class_ = 'css-aqsd8m').text)
            except AttributeError:
                price.append('NA')
            try:
                title.append(soup.find('h1', class_ = 'css-t9du53').text)
            except AttributeError:
                title.append('NA')
            try:
                seller.append(soup.find('a', class_ = 'css-1n8curp').text)
            except AttributeError:
                seller.append('NA')
            try:
                loc.append((soup.find('span',attrs={'data-testid' : 'lblPDPFooterLastOnline'}).text).split('\xa0')[0])
            except AttributeError:
                loc.append('NA')
            try:
                sell = re.search('\d+', soup.find('span',attrs={'data-testid' : 'lblPDPDetailProductSuccessRate'}).text).group(0)
                sold.append(int(sell))
            except AttributeError:
                sold.append(0)
            try:
                stok = re.search('\d+', soup.find('p',attrs={'data-testid' : 'lblPDPDescriptionProduk'}).text).group(0)
                stock.append(int(stok))
            except AttributeError:
                stock.append(1)  
            try:
                json1 = json.loads(soup.findAll('script', type = "application/ld+json")[1].string, strict = False)
                desc.append(json1['descprition'])
            except IndexError:
                desc.append('NA')
            for i in soup.findAll('img', class_ = 'success'):
                img.append(i.get('src').replace('200-square', '700'))
            img_url.append(img)
            img = []
            pbar1.update(1)
            #sleep(np.random.randint(1,5))
    data = pd.DataFrame({'URL':links, 'Item number': item, 'Price':price, 'Title': title, 'Seller Location':loc, 'Store Url': store, 'Seller ID': seller, 'Sold':sold, 'Stock': stock, 'Description': desc, 'Img Url': img_url})
    data['Img Url'] = data['Img Url'].apply(remove_char)
    data.to_excel( output_folder + '\\' + ofile + '.xlsx', index=False)
    data.to_csv( output_folder + '\\' + ofile + '.csv', index=False, encoding="utf-8-sig")
    print('Demo of the file.')
    return data.head()

def tokopedia_screenshot():
    import pandas as pd
    from time import sleep
    from tqdm.notebook import tqdm
    from selenium import webdriver
    from selenium.webdriver.common.keys import Keys
    from selenium.common.exceptions import TimeoutException, WebDriverException
    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.183 Safari/537.36'
    options = webdriver.ChromeOptions()
    options.headless = True
    options.add_argument(f'user-agent={user_agent}')
    print('This Code is created by Abhishek Chauhan. Feel free to reach in case of any problem.')
    print('-'*60)
    input_folder = input('Enter the path of folder where excel file is stored')
    print('-'*60)
    file = input('Enter file name with .xlsx & file must have a header example(url/URL)')
    print('-'*60)
    output_folder = input('Enter the path of folder where you want screenshots')
    print('-'*60)
    print('Starting capturing screenshots')
    sleep(2)
    data = pd.read_excel(input_folder +"\\" + file)  
    with tqdm(total=len(data), desc = 'Progress') as pbar:
        #with webdriver.Chrome(r"C:\chromedriver_win32\chromedriver.exe",options=options) as driver:
        for link in data.iloc[:,0]:
            str1 = link.split('://')[1]
            filename = str1.replace('/','__')
            image = filename.replace('.','_') + ".png"
            driver = webdriver.Chrome(r"C:\chromedriver_win32\chromedriver.exe",options=options)
            try:
                driver.get(link)
                sleep(1.5)
            except TimeoutException:
                print('Waiting for 1 minutes so connection could re-establish.')
                sleep(60)
                driver.get(link)
                sleep(1.5)
            for i in range(4): 
                driver.find_element_by_tag_name('body').send_keys(Keys.PAGE_DOWN)
                sleep(1)
            driver.find_element_by_tag_name('body').send_keys(Keys.HOME)
            S = lambda X: driver.execute_script('return document.body.parentNode.scroll'+X)
            driver.set_window_size(S('Width'),S('Height')) 
            try:
                driver.find_element_by_tag_name('body').screenshot(output_folder+"\\"+image)
            except TimeoutException:
                print(link + ' is not saved')
            pbar.update(1)
            driver.quit()
    print('All Screenshots are done. Kindly check your folder')
    
def tokopedia_status_checker():
    import re
    import sys
    import json
    import requests
    import numpy as np
    import pandas as pd
    from time import sleep
    from bs4 import BeautifulSoup
    from tqdm.notebook import tqdm
    from requests.exceptions import ConnectionError, ReadTimeout
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.61 Safari/537.36'}

    print('This Code is created by Abhishek Chauhan. Feel free to reach in case of any problem.')
    print('-'*60)

    input_folder = input('Enter the path of folder where excel file is stored.')
    print('-'*60)

    ifile = input('Enter file name with .xlsx & file must have a header example(url/URL)')
    print('-'*60)

    output_folder = input('Enter the path of folder where you want output file.  ')
    print('-'*60)

    ofile = input('Enter the name of output file.  ')
    print('-'*60)

    def connected_or_not():
        try:
            requests.get('http://google.com')
            return 'Connected'
        except:
            return 'Not Connected'

    links = []
    title = []

    dt = pd.read_excel(input_folder +"\\" + ifile)

    print('Starting checking status.')
    print('-'*60) 

    with tqdm(total=len(dt), desc = 'Progress') as pbar1:
        for link in dt.iloc[:,0]:
            try:
                r1 = requests.get(link, headers = headers, timeout = 20)
            except (ConnectionError, ReadTimeout):
                print('Waiting for 2 minutes so connection could re-establish.')
                sleep(120)
                try:
                    r1 = requests.get(link, headers = headers, timeout = 20)
                except (ConnectionError, ReadTimeout):
                    if connected_or_not() == 'Connected':
                        sys.exit('YOU ARE BLOKED BY TOKOPEIDA. TRY AFTER SOME TIME.')
                    elif connected_or_not() == 'Not Connected':
                        while connected_or_not() == 'Not Connected':
                            connected_or_not()
                            print('No Internet', end = '\r')
                            sleep(15)
                        print('Internet is back. Profiling again.')
                        r1 = requests.get(link, headers = headers, timeout = 20)
            soup = BeautifulSoup(r1.content, features="lxml")
            links.append(r1.url.split("?")[0])
            try:
                title.append(soup.find('h1', class_ ="css-x7lc0h").text)
            except AttributeError:
                title.append('NA')
            pbar1.update(1)
            #sleep(np.random.randint(1,5))
    data = pd.DataFrame({'URL':links, 'Status': title})
    data['Status'] = ['Removed' if x =='NA' else 'Active' for x in data['Status']]
    data.to_excel( output_folder + '\\' + ofile + '.xlsx', index=False)
    data.to_csv( output_folder + '\\' + ofile + '.csv', index=False, encoding="utf-8-sig")
    print('Demo of the file.')
    return data.head()

def lazada_screenshot():
    import pandas as pd
    from time import sleep
    from tqdm.notebook import tqdm
    from selenium import webdriver
    from selenium.webdriver.common.keys import Keys
    from selenium.common.exceptions import TimeoutException, WebDriverException
    options = webdriver.ChromeOptions()
    options.headless = True
    print('This Code is created by Abhishek Chauhan. Feel free to reach in case of any problem.')
    print('-'*60)
    input_folder = input('Enter the path of folder where excel file is stored')
    print('-'*60)
    file = input('Enter file name with .xlsx & file must have a header example(url/URL)')
    print('-'*60)
    output_folder = input('Enter the path of folder where you want screenshots')
    print('-'*60)
    print('Starting capturing screenshots')
    sleep(2)
    data = pd.read_excel(input_folder +"\\" + file)  
    with tqdm(total=len(data), desc = 'Progress') as pbar:
        with webdriver.Chrome(r"C:\chromedriver_win32\chromedriver.exe",options=options) as driver:
            for link in data.iloc[:,0]:
                i = link.split('-')[-2]
                s = link.split('-')[-1].split('.')[0]
                image = i + '-' + s + '.png'
                try:
                    driver.get(link)
                    sleep(1.5)
                except TimeoutException:
                    print('Waiting for 2.5 minutes so connection could re-establish.')
                    sleep(150)
                    driver.get(link)
                    sleep(1.5)
                driver.find_element_by_tag_name('body').send_keys(Keys.END)
                sleep(1.5)
                driver.find_element_by_tag_name('body').send_keys(Keys.HOME)
                S = lambda X: driver.execute_script('return document.body.parentNode.scroll'+X)
                driver.set_window_size(S('Width'),S('Height'))
                try:
                    driver.find_element_by_tag_name('body').screenshot(output_folder+"\\"+image)
                except WebDriverException:
                    print(link + ' is not saved')
                pbar.update(1)
    print('All Screenshots are done. Kindly check your folder')

def bukalapak_screenshot():
    import pandas as pd
    from time import sleep
    from tqdm import tqdm_notebook as tqdm
    from selenium import webdriver
    from selenium.webdriver.common.keys import Keys
    from selenium.common.exceptions import TimeoutException, WebDriverException
    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.183 Safari/537.36'
    options = webdriver.ChromeOptions()
    options.headless = True
    options.add_argument(f'user-agent={user_agent}')
    print('This Code is created by Abhishek Chauhan. Feel free to reach in case of any problem.')
    print('-'*60)
    input_folder = input('Enter the path of folder where excel file is stored')
    print('-'*60)
    file = input('Enter file name with .xlsx & file must have a header example(url/URL)')
    print('-'*60)
    output_folder = input('Enter the path of folder where you want screenshots')
    print('-'*60)
    print('Starting capturing screenshots')
    sleep(2)
    data = pd.read_excel(input_folder +"\\" + file)  
    with tqdm(total=len(data), desc = 'Progress') as pbar:
        with webdriver.Chrome(r"C:\chromedriver_win32\chromedriver.exe",options=options) as driver:
            for link in data.iloc[:,0]:
                str1 = link.split('/')[-1]
                filename = str1.split('-')[0]
                image = filename + ".png"
                try:
                    driver.get(link)
                    sleep(1.5)
                except TimeoutException:
                    print('Waiting for 2.5 minutes so connection could re-establish.')
                    sleep(150)
                    driver.get(link)
                    sleep(1.5)
                driver.find_element_by_tag_name('body').send_keys(Keys.END)
                sleep(1)
                driver.find_element_by_tag_name('body').send_keys(Keys.HOME)
                S = lambda X: driver.execute_script('return document.body.parentNode.scroll'+X)
                driver.set_window_size(S('Width'),S('Height')) 
                try:
                    driver.find_element_by_tag_name('body').screenshot(output_folder+"\\"+image)
                except TimeoutException:
                    print(link + ' is not saved')
                pbar.update(1)
                #driver.quit()
    print('All Screenshots are done. Kindly check your folder')

def facebook_screenshot():
    import pandas as pd
    from time import sleep
    from tqdm import tqdm_notebook as tqdm
    from selenium import webdriver
    from selenium.webdriver.common.keys import Keys
    from selenium.common.exceptions import TimeoutException, WebDriverException
    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.183 Safari/537.36'
    options = webdriver.ChromeOptions()
    options.headless = False
    options.add_argument(f'user-agent={user_agent}')
    print('This Code is created by Abhishek Chauhan. Feel free to reach in case of any problem.')
    print('-'*60)
    input_folder = input('Enter the path of folder where excel file is stored')
    print('-'*60)
    file = input('Enter file name with .xlsx & file must have a header example(url/URL)')
    print('-'*60)
    output_folder = input('Enter the path of folder where you want screenshots')
    print('-'*60)
    print('Starting capturing screenshots')
    sleep(2)
    data = pd.read_excel(input_folder +"\\" + file)  
    with tqdm(total=len(data), desc = 'Progress') as pbar:
        with webdriver.Chrome(r"C:\chromedriver_win32\chromedriver.exe",options=options) as driver:
            for link in data.iloc[:,0]:
                str1 = link.split('://')[1]
                filename = str1.replace('/','_')
                image = filename.replace('.','_') + ".png"
                try:
                    driver.get(link)
                    sleep(1.5)
                    driver.maximize_window()
                    sleep(1.5)
                except TimeoutException:
                    print('Waiting for 2.5 minutes so connection could re-establish.')
                    sleep(150)
                    driver.get(link)
                    sleep(2)
                S = lambda X: driver.execute_script('return document.body.parentNode.scroll'+X)
                driver.set_window_size(S('Width'),S('Height')) 
                try:
                    driver.find_element_by_tag_name('body').screenshot(output_folder+"\\"+image)
                except TimeoutException:
                    print(link + ' is not saved')
                pbar.update(1)
                #driver.quit()
    print('All Screenshots are done. Kindly check your folder')
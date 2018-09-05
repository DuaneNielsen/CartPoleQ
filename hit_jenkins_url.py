import requests
headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
r = requests.get(headers=headers, url='http://desktop-43a62tm:8080/job/Most%20Improved/build?token=fbf4e997369f4c10fbae201d28d18e55')
print(r.status_code)
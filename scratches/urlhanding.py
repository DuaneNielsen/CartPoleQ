from urllib.parse import quote

url = r'runs\jenkins-View-17\d68fd69157daa29d35dc44d4dfb8a168047305e0\atariconv_v6-filter_stack-32-64-256-256-256'

url = url.replace('\\', '\\\\')

url = quote(url)

url = 'http://lt3vtgjm2:6006/#scalars&regexInput=' + url

print(url)


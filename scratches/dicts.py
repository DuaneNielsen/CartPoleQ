
dict = {}

def increment():
    if 'key' not in dict:
        dict['key'] = 1
    else:
        dict['key'] += 1
    print(dict['key'])

increment()
increment()
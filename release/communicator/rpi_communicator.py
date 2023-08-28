import urllib3
import json

http = urllib3.PoolManager()

def getAction(t,h,a):
    url = 'http://127.0.0.1:8000/action?' + 'query=' + str(t) + '&query=' + str(h) + '&query=' + str(a)
    res = http.request('GET', url)
    d = res.data.decode('utf-8')
    act = json.loads(d)
    print(act['action'])
    return act['action']

getAction(34.56, 298.9,349.9)
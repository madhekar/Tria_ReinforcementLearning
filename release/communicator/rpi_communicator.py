import urllib3
import json

class RPL():
    def __init__(self, _http=None, _baseUrl=None):
        self.http = _http
        self.baseUrl = _baseUrl

    def getAction(self, t,h,a):
        url = self.baseUrl + 'query=' + str(t) + '&query=' + str(h) + '&query=' + str(a)
        res = self.http.request('GET', url)
        d = res.data.decode('utf-8')
        return json.loads(d)['action']

if __name__ == '__main__':    
    '''
    define constant server base url & urllib3 PoolManager
    '''
    baseUrl = 'http://127.0.0.1:8000/action?'
    http = urllib3.PoolManager()
    rpl = RPL(http, baseUrl)
   
    '''
    call RPI getAction
    '''
    print('action: ', rpl.getAction(34.56, 298.9,349.9))
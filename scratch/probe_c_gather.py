from blackbox_api import BlackBoxAPI
import re

def get_expected(api, text):
    res = api.submit(text, 'dummy')
    hint = res.get('hint', '')
    m = re.search(r"payload is '(.*?)'", hint)
    return m.group(1) if m else ''

api = BlackBoxAPI()
for word in ['a', 'ab', 'abc', 'test']:
    print('HINT:', word, '->', get_expected(api, word))
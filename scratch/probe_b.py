from blackbox_api import BlackBoxAPI
import re

def get_expected(api, text):
    res = api.submit(text, 'dummy')
    hint = res.get('hint', '')
    m = re.search(r"payload is '(.*?)'", hint)
    return m.group(1) if m else ''

api = BlackBoxAPI()
# Query reference word 'a'
encoded_a = get_expected(api, 'a')
# 'a' is ASCII 97. If encoded_a is 'h' (ASCII 104), shift offset is 104 - 97 = 7.
offset = ord(encoded_a) - 97 if encoded_a else 7

def encode(text, offset_val):
    shifted = []
    for i, c in enumerate(text):
        shifted.append(chr((ord(c) + offset_val + i) % 256))
    return ''.join(shifted)[::-1]

payload = encode('HbolMJ', offset)
print('SOLVED_PAYLOAD:', payload)
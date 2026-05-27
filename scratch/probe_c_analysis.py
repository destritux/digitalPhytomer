def encode(text, offset_val):
    shifted = []
    for i, c in enumerate(text):
        shifted.append(chr((ord(c) + offset_val + i) % 256))
    return ''.join(shifted)[::-1]

# Shift offset: 'a' -> 'h' is shift of 7
offset = 7
payload = encode('HbolMJ', offset)
print('SOLVED_PAYLOAD:', payload)
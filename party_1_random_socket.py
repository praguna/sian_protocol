import socket, BNAuth as auth
from utils import Party
from biometric_template import extractandenconding
from sys import argv
import numpy as np, json

# '''
# This is the client side
# '''

key_default = 'key_2048'

with open('random_client_key.json', 'r') as f:
     k = json.load(f)[key_default]
     R = np.array([int(e) for e in k])


if __name__=='__main__':
    img_path='tests/p1_1.bmp'
    template, mask, _ = extractandenconding.extractFeature(img_path, radial_resolution=16, angular_resolution=64)
    X = template.flatten().astype(int)
    M = mask.flatten().astype(int)
    N = np.zeros_like(X) 
    p1 = np.argwhere(M == 1)
    N[p1] = X[p1]
    X = np.logical_xor(X, R).astype(int)
    N = np.logical_xor(N, R).astype(int)
    noise, change = float(argv[2]), float(argv[3])
    noise_pos = np.random.choice(range(len(X)), int(noise * len(N)), False)
    for i in noise_pos: N[i]^=1
    noise_pos = np.random.choice(range(len(X)), int(change * len(X)), False)
    for i in noise_pos: X[i]^=1
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(('0.0.0.0', int(argv[1]))) 
    client.send(bytes('-', encoding="utf-8"))
    
    bnAuth = auth.BNAuth(X ,M, N, Party.P1, client, direct_index=False)
    try:
        a = bnAuth.perform_secure_match()
    except Exception as e:
        print(f'{e}: Identical noisy image rejected by the Server')
        a = 0
    client.close()

    k = int(argv[4]) if len(argv) == 5 else 0
    j = 0
    while a > 1 and  j < k:
        print(f'>50% assumption attempt {j+1}')
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect(('0.0.0.0', int(argv[1])))
        client.send(bytes('-', encoding="utf-8"))
        bnAuth = auth.BNAuth(X ,M, N, Party.P1, client, direct_index=False)
        a = bnAuth.perform_secure_match()
        client.close()
        j+=1


    if a > 1 and k > 0:
        print('Using 75% constraint assumption')
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect(('0.0.0.0', int(argv[1])))
        client.send(bytes(':', encoding="utf-8"))
        bnAuth = auth.BNAuth(X ,M, N, Party.P1, client, direct_index=False)
        a = bnAuth.perform_secure_match(True)
        client.close()
    
    print('Kill the server terminal with Ctrl+C / run again for another comparison')
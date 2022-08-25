
import socket, BNAuth_parallel as auth
from utils import Party
from biometric_template import extractandenconding
import time, os
from sys import argv
import numpy as np

# '''
# This is the client side
# '''

if __name__=='__main__':
    img_path='tests/p1_1.bmp'
    template, mask, _ = extractandenconding.extractFeature(img_path, radial_resolution=40, angular_resolution=64)
    X = template.flatten().astype(int)
    M = mask.flatten().astype(int)
    noise, change = float(argv[2]), float(argv[3])
    noise_pos = np.random.choice(range(len(X)), int(noise * len(X)), False)
    for i in noise_pos: M[i]^=1
    noise_pos = np.random.choice(range(len(X)), int(change * len(X)), False)
    for i in noise_pos: X[i]^=1
    client = [socket.socket(socket.AF_INET, socket.SOCK_STREAM) for _ in range(4)]
    for x in client: x.connect(('0.0.0.0', int(argv[1]))) 
    client[0].send(bytes('-', encoding="utf-8"))
    
    bnAuth = auth.BNAuth(X , M, Party.P1, client, direct_index=False)
    try:
        a = bnAuth.perform_secure_match()
    except Exception as e:
        print(f'{e}: Identical noisy image rejected by the Server')
        a = 0
    for x in client: x.close()
    
    print('Kill the server terminal with Ctrl+C / run again for another comparison')
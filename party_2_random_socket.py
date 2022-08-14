import socket, numpy as np, BNAuth as auth
from utils import Party
import scipy.io as sio, time, os
from biometric_template import extractandenconding
from sys import argv
import argparse, json


# '''
# This is the server side
# '''

key_default = 'key_2048'

with open('random_client_key.json', 'r') as f:
     k = json.load(f)[key_default]
     R = np.array([int(e) for e in k])


if __name__=='__main__':
    serv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serv.bind(('0.0.0.0', int(argv[1])))
    serv.listen()
    print('server started !!!')
    img_path='tests/p1_1.bmp'
    A = []
    while True:
        try:
            conn, addr = serv.accept()
            mode = conn.recv(8096).decode('utf-8')
            b = False
            # running mode : 25% or 75% assumption
            if mode.find(':') != -1: b = True
            X, M, _ = extractandenconding.extractFeature(img_path,  radial_resolution=16, angular_resolution=64)
            X, M = X.flatten(), M.flatten()
            N = np.zeros_like(X) 
            p1 = np.argwhere(M == 1)
            N[p1] = X[p1]
            X = np.logical_xor(X, R).astype(int)
            N = np.logical_xor(N, R).astype(int)
            bnAuth = auth.BNAuth(X ,M, N ,Party.P2, conn, direct_index=False)
            s = time.time()
            a = bnAuth.perform_secure_match(b)
            e = time.time()
            # print(f'time taken : {e-s}s')
            A.append(e-s)
            print(sum(A) / len(A))
            f = False
            if a < 0: 
                f = True
                print('distance invalid (hd > 1) setting it 1')
            a =  min(a, 1)
            print(f'hd : {a}')
            if a > 0.3: print('Reject with 0.3 thershold')
            else: print('Decision : Accept (0.3 thershold)')
        except Exception as e:
            print(e)
        finally: conn.close()
    serv.close()
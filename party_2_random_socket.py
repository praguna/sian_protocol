import socket, numpy as np, BNAuth_parallel as auth
from utils import Party
import scipy.io as sio, time, os
from biometric_template import extractandenconding
from sys import argv


# '''
# This is the server side
# '''



if __name__=='__main__':
    serv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serv.bind(('0.0.0.0', int(argv[1])))
    serv.listen()
    print('server started !!!')
    img_path='tests/p1_1.bmp'
    while True:
        try:
            conn = []
            for _ in range(4):
                s , addr = serv.accept()
                conn.append(s)
            mode = conn[0].recv(8096).decode('utf-8')
            b = False
            # running mode : 25% or 75% assumption
            if mode.find(':') != -1: b = True
            X, M, _ = extractandenconding.extractFeature(img_path,  radial_resolution=40, angular_resolution=64)
            bnAuth = auth.BNAuth(X.flatten() , M.flatten(), Party.P2, conn, direct_index=False)
            s = time.time()
            a = bnAuth.perform_secure_match(b)
            e = time.time()
            print(f'time taken : {e-s}s')
            f = False
            if a < 0: 
                f = True
                print('distance invalid (hd > 1) setting it 1')
            a =  min(a, 1)
            print(f'hd : {a}')
            if a > 0.3: print('Reject with 0.3 thershold')
            else: print('Decision : Accept (0.3 thershold)')
            break
        except Exception as e:
            print(e)
        finally: 
            for x in conn: x.close()
    serv.close()
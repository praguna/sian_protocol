from pickletools import int4
from time import time
from utils import Party, M as AND_MATRIX, and_on_numpy2D, convert_to_str, xor_on_numpy1D, xor_on_numpy2D, split_by_curly, serialize_nd_array
import numpy as np
import socket, json, tqdm
from collections import deque
import os, struct

# setting the seed 
# np.random.seed(42)
oc1 , oc2 =  np.array([0, 1 ,1, 0]), np.array([0, 1, 0, 0])
class BNAuth(object):
    '''
    Implementation of Biometric matching with noise without exchanging templates
    '''

    def __init__(self, X, M, N, party_type : Party, socket: socket.socket = None, direct_index = False ,error_rate = 0.25) -> None:
        '''
        arg X -> template
        arg MI -> masked indices
        arg party_addr -> other party's addr
        '''
        self.X = X.astype(np.int8)
        self.M = M.astype(np.int8)
        self.N = N.astype(np.int8)
        self.d = len(self.X)
        self.error_rate = error_rate
        self.m = min(len(self.N) // 4, 100)
        self.party_type = party_type 
        self.socket = socket
        self.message_queue = deque()
        self.n = int(0.90 * self.m)
        self.num_dist = 1
        self.selected_octect_index = None
        self.T = []
        self.visited_mask_pos = np.zeros_like(self.N)
        self.count  = 0
        assert self.m > 0
    

    def send_to_peer(self, data):
        '''
        sends it to the peer socket
        '''
        self.socket.sendall(bytes(data, encoding="utf-8"))
        # print(f'sent : {data}')

    
    def recieve_from_peer(self, num_bytes = 126000):
        '''
        decode and extract data from the peer
        '''
        if len(self.message_queue) > 0:
            v = self.message_queue.popleft()
            return json.loads(v)
        msg = self.socket.recv(num_bytes).decode('utf-8')
        self.message_queue.extend(split_by_curly(msg))
        self.count += len(msg)
        # print(f'received : {self.message_queue}')
        return json.loads(self.message_queue.popleft())


    def recieve_has_expected_noise(self, pos : np.ndarray, masked_xor)->bool:
        '''
        P2 sends True if expected noise is present and verified
        '''
        self.send_to_peer(json.dumps({'pos' : serialize_nd_array(pos), 'masked_xor' : int(masked_xor)}))
        has_expected_noise = self.recieve_from_peer()
        return has_expected_noise['has_expected_noise']

    
    '''
    Assuming uncertain position MI the same for a person
    '''
    def get_masked_bit_quad(self):
        '''
        receive / send corresponding 4 masked bit positions, compare xor difference
        '''
        # Note : for testing just append pos beside the result
        def P1():
            is_expected_noise = False
            while not is_expected_noise:
                pos = np.random.choice(range(self.d), 4, False) 
                is_expected_noise  = self.recieve_has_expected_noise(pos, xor_on_numpy1D(self.N[pos]))
            return self.N[pos] # implementing only for p1
        
        def P2():
            s = time()
            is_expected_noise = False
            while not is_expected_noise:
                data = self.recieve_from_peer()
                if xor_on_numpy1D(self.N[data['pos']]) != data['masked_xor'] and np.sum(self.visited_mask_pos[data['pos']]) < 2: is_expected_noise = True
                if is_expected_noise: self.visited_mask_pos[data['pos']] = 1
                self.send_to_peer(json.dumps({'has_expected_noise' :  is_expected_noise}))
                t = time()
                if t - s > 3 : raise Exception("Timing out after 3 secs")
            return self.N[data['pos']] # implementing only for p1

        return P1() if self.party_type == Party.P1 else P2()


    def preprocess(self):
        '''
        creates m octects in the party
        '''
        return np.array([self.get_masked_bit_quad() for _ in range(self.m)])
        

    def create_distributed_vectors(self, X):
        '''
        create 2 vectors as : X = X1 xor X2 and distribute X2 over to party_addr
        '''
        X1 = np.array([np.random.randint(0, 2) for _ in range(self.d)])
        X2 = xor_on_numpy2D([X1, X])
        self.send_to_peer(json.dumps({"Y1" : serialize_nd_array(X2)}))
        return (X1, X2)


    def fetch_distributed_vector(self):
        '''
        fetch vector Y1 from party_addr
        '''
        return self.recieve_from_peer()['Y1']

    def perform_secure_xor(self, X, Y):
        '''
        returns secure xor between X and Y
        '''
        return np.logical_xor(X, Y)

    def dot_product(self, octect : np.ndarray):
        '''
        return 1x3 vectors joined into a list for further usage 
        '''
        return octect.T.astype(bool) @ AND_MATRIX

    def send_xors_over(self, x1_xor_a1, y1_xor_b1):
        '''
        sends x1 xor a1 and y1 xor b1 to another party
        '''
        self.send_to_peer(json.dumps({'xors' : (bool(x1_xor_a1), bool(y1_xor_b1))}))

    def recieve_xors(self):
        '''
        recieve xors from the other party i.e :  x2 xor a2 and y2 xor b2
        '''
        a = self.recieve_from_peer()
        return a['xors']

    def compute_Z(self, XA, YB, x, y, c) -> np.ndarray:
        '''
        computes z1 or z2 based on the party type
        '''
        if self.party_type == Party.P1: 
            return  (XA & YB) ^ (x & YB) ^ (y & XA) ^ c
        return (x & YB) ^ (y & XA) ^ c

    def fetch_octect(self):
        '''
        gets an octect for each party
        '''
        def P1():
            idx = np.random.choice(list(self.selected_octect_index), 1)
            self.send_to_peer(json.dumps({'idx' : serialize_nd_array(idx)}))
            return idx
        def P2():
            idx = self.recieve_from_peer()['idx']
            return idx
        idx = P1() if self.party_type == Party.P1 else P2()
        octect = self.octects[idx]
        return octect.ravel()
    
    def fetch_octect_bulk(self, batch_size = 128):
        '''
        gets octects for each party
        '''
        def P1():
            idx = np.random.choice(list(self.selected_octect_index), batch_size)
            self.send_to_peer(json.dumps({'idx' : serialize_nd_array(idx)}))
            return idx
        def P2():
            idx = self.recieve_from_peer()['idx']
            return idx
        idx = P1() if self.party_type == Party.P1 else P2()
        return idx

    def perform_computation_phase(self, octect, x, y):
        '''
        finds out partial And (Z1)
        '''
        a1, b1, c1 = self.dot_product(octect).astype(np.int8)
        x1_xor_a1, y1_xor_b1 = xor_on_numpy2D([a1, x]) , xor_on_numpy2D([b1, y])
        self.send_xors_over(x1_xor_a1, y1_xor_b1)
        x2_xor_a2, y2_xor_b2  = self.recieve_xors()
        XA = xor_on_numpy2D([x1_xor_a1, x2_xor_a2])
        YB = xor_on_numpy2D([y , b1 , y2_xor_b2])
        Z = self.compute_Z(XA, YB, x, y, c1)
        return Z

    def perform_computation_phase_v2(self, octect, w, v, noise = False):
        '''
        optimized without util function calls
        '''
        a1, b1, c1 = octect[2] ^ octect[3] , octect[1] ^ octect[3] , octect[3] if not noise else  octect[0] ^ octect[1] ^ octect[2]
        x1_xor_a1, y1_xor_b1 = a1^w , b1^v
        self.send_xors_over(x1_xor_a1, y1_xor_b1)
        x2_xor_a2, y2_xor_b2  = self.recieve_xors()
        XA = x1_xor_a1 ^ x2_xor_a2
        YB = y1_xor_b1 ^ y2_xor_b2
        Z =  (XA & YB) ^ (w & YB) ^ (v & XA) ^ c1  if self.party_type == Party.P1 else (w & YB) ^ (v & XA) ^ c1 
        return Z

    def perform_computation_phase_v3(self, octect, w, v, noise = False):
        '''
        optimized without util function calls
        '''
        a1, b1, c1 = octect[2] ^ octect[3] , octect[1] ^ octect[3] , octect[3] if not noise else  octect[0] ^ octect[1] ^ octect[2]
        x1_xor_a1, y1_xor_b1 = a1^w , b1^v
        self.socket.sendall(struct.pack('??',x1_xor_a1, y1_xor_b1))
        s = self.socket.recvmsg(2)[0]
        x2_xor_a2, y2_xor_b2  = struct.unpack('??',s)
        self.count+=2
        XA = x1_xor_a1 ^ x2_xor_a2
        YB = y1_xor_b1 ^ y2_xor_b2
        Z =  (XA & YB) ^ (w & YB) ^ (v & XA) ^ c1  if self.party_type == Party.P1 else (w & YB) ^ (v & XA) ^ c1 
        return Z
    
    def calculate_sum(self, W1, W2):
        '''
        perform complete sum as :
        W = (w11 xor w12).......()
        '''
        bits = xor_on_numpy2D([W1, W2])
        return int(convert_to_str(bits), 2)


    def perform_distillation(self, X1, Y1):
        '''
        get the final octect after elimination
        '''
        assert len(self.octects) >= 0
        if len(self.octects) == 1 : return self.octects[0]


        def P1():
            octect_set = {i : e  for i, e in enumerate(self.octects)}
            for _ in range(self.n):
                if len(octect_set) < 2: break
                # send / fetch indices
                j, match = 0 , True
                indices = np.random.choice(list(octect_set.keys()), 2, False)
                self.send_to_peer(json.dumps({'indices' : serialize_nd_array(indices)}))
                idx = np.random.choice(self.d, self.num_dist, False)
                while j < self.num_dist and match:
                    # ind = np.random.choice(self.d, 1)
                    ind = [idx[j]]
                    x, y = X1[ind], Y1[ind]
                    self.send_to_peer(json.dumps({'pos' : serialize_nd_array(ind)}))
                    A = self.perform_computation_phase_v2(octect_set[indices[0]],x, y)
                    B = self.perform_computation_phase_v2(octect_set[indices[-1]],x, y)
                    z1 = A ^ B
                    # fetch/send z2 / z1 from the other party
                    self.send_to_peer(json.dumps({'z2' : serialize_nd_array(z1)}))
                    z2 = self.recieve_from_peer()['z2']
                    match = z1 == z2
                    j+=1
                if match: # compare z1 and z2
                    del_idx = np.random.choice(indices, 1)
                    self.send_to_peer(json.dumps({'del_idx' : serialize_nd_array(del_idx)}))
                    octect_set.pop(del_idx[0])
                else:
                    for e in indices: octect_set.pop(e)
            return sorted(list(octect_set.keys()))
        
        def P2():
            octect_set = {i : e  for i, e in enumerate(self.octects)}
            for _ in range(self.n):
                if len(octect_set) < 2: break
                j, match = 0 , True
                indices = self.recieve_from_peer()['indices']
                while j < self.num_dist and match:
                    ind = self.recieve_from_peer()['pos']
                    x, y = X1[ind], Y1[ind]
                    A = self.perform_computation_phase_v2(octect_set[indices[0]], x, y)
                    B = self.perform_computation_phase_v2(octect_set[indices[-1]], x, y)
                    z1 = A ^ B
                    # fetch/send z2 / z1 from the other party
                    self.send_to_peer(json.dumps({'z2' : serialize_nd_array(z1)}))
                    z2 = self.recieve_from_peer()['z2']
                    match = z1 == z2
                    j+=1
                if match: # compare z1 and z2
                    del_idx = self.recieve_from_peer()['del_idx']
                    octect_set.pop(del_idx[0])
                else:
                    for e in indices: octect_set.pop(e)
            return sorted(list(octect_set.keys()))
        
        return P1() if self.party_type == Party.P1 else P2()
    
    
    def test_secure_and_xor(self, Z1):
        '''
        method to verify both the operations work, use before distillation phase in secure matching
        '''
        X = self.X
        self.send_to_peer(json.dumps({'X' : serialize_nd_array(X)}))
        Y = self.recieve_from_peer()['X']
        a = []
        octect = np.array([1, 0, 0, 1]) if self.party_type == Party.P1 else np.array([0, 1, 0, 1])
        for i in range(self.d):
            w, v = self.X1[i], self.Y1[i]
            # m = self.perform_computation_phase(oc.flatten(), x, y)
            Z = self.perform_computation_phase_v2(octect, w, v) 
            a.append(Z)
        self.send_to_peer(json.dumps({'xor' : serialize_nd_array(a)}))
        b = self.recieve_from_peer()['xor']
        
        self.send_to_peer(json.dumps({'Z' : serialize_nd_array(Z1)}))
        Z2 = self.recieve_from_peer()['Z']
        assert np.all(np.logical_xor(Z2, Z1).astype(int) == np.logical_xor(X, Y))
        print('passed xor!!')

        assert np.all(np.logical_xor(a, b).astype(int) == np.logical_and(X, Y).astype(int))
        print('passed and!!')
        exit(0)
    

    def not_mask_secure_and(self, noise = False):
        '''
        returns not(M1) ^ not(M2) 
        '''
        M_X1, _ = self.create_distributed_vectors(np.logical_not(self.M).astype(np.int8))
        M_Y1 = np.array(self.fetch_distributed_vector())
        a = []
        if self.party_type == Party.P2: M_X1, M_Y1 = M_Y1, M_X1
        while self.selected_octect_index == None or len(self.selected_octect_index) == 0:
            self.selected_octect_index = self.perform_distillation(M_X1, M_Y1)
        x = time()
        batch, k = 128, 0
        octets = self.fetch_octect_bulk(batch)
        for i in range(self.d):
            if k == batch: 
                octets = self.fetch_octect_bulk(batch)
                k = 0
            w, v = M_X1[i], M_Y1[i]
            Z = self.perform_computation_phase_v2(self.octects[octets[k]], w, v, noise)
            k+=1
            a.append(Z) 
        self.send_to_peer(json.dumps({'xor' : serialize_nd_array(a)}))                        
        b = self.recieve_from_peer()['xor']
        y = time()
        # print(y - x)
        return np.logical_xor(a, b).astype(np.int8)


    def perform_secure_match(self, noise = False):
        '''
        runs secure matching algorithm on both the parties P1 and P2 independently
        '''
        self.octects = self.preprocess()

        R = self.not_mask_secure_and(noise)
        self.X = np.logical_and(R, self.X).astype(np.int8)

        self.X1, _ = self.create_distributed_vectors(self.X)
        self.Y1 = np.array(self.fetch_distributed_vector())
        if self.party_type == Party.P2: self.X1, self.Y1 = self.Y1, self.X1
        Z1 = self.perform_secure_xor(self.X1, self.Y1)


        self.message_queue.clear()
        s, batch = 0, 128
        lt = int(np.log2(batch)) 
        for k in range(0, self.d, batch):
            W = [Z1[k]]
            v =  None
            octets = self.fetch_octect_bulk(batch)
            for i in range(batch-1):
                v = Z1[i+1+k]
                j = len(W) - 1
                while j>=0:  
                    w = W[j]
                    f = w^v
                    v = self.perform_computation_phase_v3(self.octects[octets[i]].ravel(), w, v, noise)
                    W[j] = f
                    j-=1  
                if len(W) <= lt: W.insert(0, v)
            self.send_to_peer(json.dumps({'W2' : serialize_nd_array(W)}))
            W2 = self.recieve_from_peer()['W2']
            s+=self.calculate_sum(W, W2)

        # print(self.count)
        tbits = self.d - np.sum(R == 0)
        return  s / tbits
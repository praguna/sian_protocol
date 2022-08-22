import numpy as np
from biometric_template import extractandenconding
from sys import argv
def hamming_distance(x1, m1, x2, m2):
        mask = np.logical_or(m1, m2)
        nummaskbits = np.sum(mask == 1)
        totalbits = x1.size - nummaskbits
        C = np.logical_xor(x1, x2)
        C = np.logical_and(C, np.logical_not(mask))
        if totalbits == 0: return 1
        return  np.sum(C == 1) / totalbits

X1, M1, _ = extractandenconding.extractFeature('tests/'+argv[1],  radial_resolution=16, angular_resolution=64)
X1 = X1.flatten()
M1 = M1.flatten()
N1 = np.zeros_like(X1) 
p1 = np.argwhere(M1 == 1)
N1[p1] = X1[p1]
X2, M2, _ = extractandenconding.extractFeature('tests/'+argv[2],  radial_resolution=16, angular_resolution=64)
X2 = X2.flatten()
M2 = M2.flatten()
N2 = np.zeros_like(X2) 
p1 = np.argwhere(M2 == 1)
N2[p1] = X2[p1]
print(f'hd : {hamming_distance(X1, M1, X2, M2)}, noise ratio : {(np.sum(np.logical_xor(N1, N2)) / len(N1))}')
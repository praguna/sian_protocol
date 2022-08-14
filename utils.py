from enum import Enum
import numpy as np
import string 

'''
 Logical truth table of AND gate with 2 inputs
'''
M = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1] ]).astype(bool)

class Party(Enum):
    P1 = 'p1'
    P2 = 'p2'

def xor_on_numpy1D(A : np.ndarray):
    '''
    Do xor on numpy bit array
    '''
    r = 0
    for e in A.flatten(): r^=e
    return r

def xor_on_numpy2D(As : np.ndarray):
    '''
    Do xor on numpy bit array
    '''
    A = np.vstack(As).T
    r = np.zeros(A.shape[0], dtype=int)
    for i in range(A.shape[0]): r[i]^=xor_on_numpy1D(A[i, :])
    return r

def and_on_numpy1D(A : np.ndarray):
    '''
    Do and on numpy bit array
    '''
    r = 1
    for e in A.flatten(): r&=e
    return r

def and_on_numpy2D(As : np.ndarray):
    '''
    Do and on numpy bit array
    '''
    A = np.vstack(As).T
    r = np.ones(A.shape[0], dtype=int)
    for i in range(A.shape[0]): r[i]&=and_on_numpy1D(A[i, :])
    return r

def convert_to_str(A : np.ndarray):
    return ''.join(list(A.astype(str)))


def remove_non_printable(ch):
    return ''.join([e for e in ch if e in string.printable])

def split_by_curly(s: str):
    s = remove_non_printable(s)
    return [e+'}' for e in s.split('}') if len(e) > 0]

def serialize_nd_array(A : np.ndarray):
    return [int(e) for e in A]


if __name__ == "__main__":
    assert xor_on_numpy1D(np.array([0, 1, 0, 1])) == 0
    assert xor_on_numpy1D(np.array([0, 1, 1, 1])) == 1
    X1 = np.array([0, 1, 0, 1])
    X2 = np.array([0, 1, 1, 1])
    assert np.all(xor_on_numpy2D([X1, X2]) == np.array([0, 0, 1, 0]))
    assert np.all(xor_on_numpy2D([0, 1]) == np.array([1]))
    assert convert_to_str(np.array([1,0,1,1])) == '1011'
    assert and_on_numpy1D(np.array([1, 1, 1, 1])) == 1
    assert np.all(and_on_numpy2D([X1, X2]) == np.array([0, 1, 0, 1]))
    assert serialize_nd_array(np.array([0, 0 , 1, 1])) == [0, 0 , 1, 1]
import numpy as np
from numba import njit
from scipy.interpolate import interp1d as interpolate
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#n = 200
#m = 50

nn1 = 1.
nn2 = 5
ss1 = 0.1
ss2 = 0.1
mc = 0.1
kk = 1
L = 1

with open("starts.txt", 'r') as f:
	s1, s2 = list(map(float, f.readline().split()[:2]))
	n, m = list(map(int, f.readline().split()[:2]))

p1 = 10
p2 = 1
#s1 = 0.5
#s2 = 0.2

@njit
def k1(s):
	if s < ss1:
		return 0
	return ((s - ss1) / (1 - ss1)) ** 2
@njit
def k2(s):
	if s > 1 - ss2:
		return 0
	return (1 - (1 - s) / ss2) ** 2

@njit
def B(s):
	return (k1(s) / nn1) + (k2(s) / nn2)

@njit
def W(s, dp):
	return -kk * B(s) * dp

@njit
def phi(s):
	return (k1(s) / nn1) / (k1(s) / nn1 + k2(s) / nn2)

@njit
def b_simple(s, dp):
	return -20 * dp * kk * s / nn1

@njit
def b_complex(s, dp):
	if s < ss1:
		return 0
	return -20 * dp * kk * ((s - ss1) / (1 - ss1)) / nn1

vec_phi = np.vectorize(phi)
vec_b_complex = np.vectorize(b_complex)

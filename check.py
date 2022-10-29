import numpy as np
from numba import njit
from scipy.interpolate import interp1d as interpolate
import matplotlib.pyplot as plt

m = 50

nn1 = 1.
nn2 = 5
ss1 = 0.1
ss2 = 0.1
mc = 0.1
kk = 1

@njit
def k1(s):
	return s ** 2

@njit
def k2(s):
	return (1 - s) ** 2

@njit
def B(s):
	return (k1(s) / nn1) + (k2(s) / nn2)

@njit
def W(s, dp):
	return (-kk * B(s) * dp)

@njit
def phi(s):
	return (k1(s) / nn1) / (k1(s) / nn1 + k2(s) / nn2)

@njit
def bb(s, dp):
	return -20 * dp * kk * s / nn1

vec_phi = np.vectorize(phi)

def s_iter(s, dp, t, h, tau):
	ww = bb(s[t], dp=dp)
	print(f"dp = {dp}, w_max = {max(ww)}")
	s[t + 1][1:] = s[t][1:] - ww[1:] * (tau / h) * (s[t][1:] - s[t][:-1])
	return s

def main():

	L = 1
	p1 = 10
	p2 = 1

	dp = (p2 - p1) / L
	
	s_pos = np.linspace(0, L, m)

	w_water = bb(s_pos, dp=dp)
	w_max = max(w_water)	
	h = L / (m - 1)
	tau =  0.5 * h  / w_max
	print(f"tau = {tau}")
	
	n = int((1 / w_max) / tau)
	print(n)
	s = np.zeros((n, m))
	s[0] = 0.2
	s[:, 0] = 0.5

	for t in range(n - 1):
		s = s_iter(s, dp, t, h, tau)
		if t < 20:
			print(f"t = {t}")
			print(s[t])
	#print(s)
	print("\n")
	return s

def graph(s):
	fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)

	fig.set_label("Graph")
	ax2.set_ylabel('Водонасыщенность')
	ax3.set_xlabel('Координаты')

#	print(n)
	n = len(s)
	x = np.linspace(0, 1, m)

	ax1.plot(x, s[0], "r")
	ax2.plot(x, s[n // 3], "g")
	ax3.plot(x, s[2 * n // 3], "b")
	ax4.plot(x, s[n - 1], "black")
	print(max(s, key=sum))
	fig.set_label("Graph")

	ax1.set_ylim([0, 1])
	ax2.set_ylim([0, 1])
	ax3.set_ylim([0, 1])
	ax4.set_ylim([0, 1])
	ax1.set_xlim([0, 1])
	ax2.set_xlim([0, 1])
	ax3.set_xlim([0, 1])
	ax4.set_xlim([0, 1])
	
	plt.show()

if __name__ == "__main__":
	s = main()
	graph(s)

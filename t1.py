import numpy as np
from numba import njit
from scipy.interpolate import interp1d as interpolate
import matplotlib.pyplot as plt

n = 5000
m = 200
NNN = int(m / 15)

nn1 = 1.
nn2 = 2.
ss1 = 0.25
ss2 = 0.2
mc = 0.1

@njit
def B(s):
	return (k1(s) / nn1) + (k2(s) / nn2) 

@njit
def k1(s):
	return s ** 2
@njit
def k2(s):
	return (1 - s) ** 2

@njit
def phi(s):
	return (k1(s) / nn1) / (k1(s) / nn1 + k2(s) / nn2)


def s_iter(s, p, t, h, tau):

	for i in range(1, m):
		if (i == m - 1):
			p_fun = interpolate(np.arange(len(p[t + 1])), p[t + 1], fill_value="extrapolate")
			p_lst = p_fun(m)
			ab1 = (p_lst - p[t + 1][i]) / h
			shp = s[t][i]
		else:
			shp = (s[t][i] + s[t][i + 1]) / 2
			ab1 = (p[t + 1][i + 1] - p[t + 1][i]) / h
		
		shm = (s[t][i - 1] + s[t][i]) / 2
		aa1 = B(shp) * phi(shp)
		aa2 = B(shm) * phi(shm)
		ab2 = (p[t + 1][i] - p[t + 1][i - 1]) / h
		
		#print(f"aa1 = {aa1},\taa2 = {aa2}")
		s[t + 1][i] = np.abs(s[t][i] + (tau / (mc * h)) * (aa1 * ab1 - aa2 * ab2))
		if (s[t + 1][i] > 1 - ss2):
			s[t + 1][i] = 1 - ss2
	return s

def progonka_count(a, b, c, p1, p2):
	y = np.zeros(m)
	alpha = np.zeros(m)
	beta = np.zeros(m)
	x = np.zeros(m)

	y[0] = b[0]
	alpha[0] = -c[0]/y[0]
	beta[0] = 0
	for i in range(1, m - 1):
		y[i] = b[i] + a[i] * alpha[i - 1]
		alpha[i] = -c[i] / y[i]
		beta[i] = -a[i] * beta[i - 1] / y[i]
	y[m - 1] = b[m - 1] + a[m - 1] * alpha[m - 2]
	beta[m - 1] = -a[m - 1] * beta[m - 2] / y[m - 1]

	x[m - 1] = p1
	for i in range(m - 2, -1, -1):
		x[i] = alpha[i] * x[i + 1] + beta[i]
	return x[::-1]

def get_abch(s, t, L):
	a, b, c = np.zeros(m), np.zeros(m), np.zeros(m)

	c[0] = B((s[t][0] + s[t][1]) / 2)
	b[0] = -2 * B(s[t][0])
	for i in range(1, m - 1):
		shm = (s[t][i - 1] + s[t][i]) / 2
		shp = (s[t][i] + s[t][i + 1]) / 2
		a[i] = B(shm)
		b[i] = -B(shm) - B(shp)
		c[i] = B(shp)
	a[m - 1] = B((s[t][m - 2] + s[t][m - 1]) / 2)
	b[m - 1] = -2 * B(s[t][m - 1])

	return a, b, c, (L / (n - 1)), (L / (m - 1))

def print_mass(mas, s):
	print(s)
	for i in reversed(mas):
		for k, j in enumerate(i):
			if k > 3 and k % NNN:
				continue
			print(f"{j:.5f}", end="  ")
		print("")

def print_onemas(mas, s):
	print(s)
	for k, j in enumerate(mas):
		if k > 3 and k % NNN:
			continue
		print(f"{j:.5f}", end="  ")
	print("")

def main():
	p1 = 100
	p2 = 1
	s = np.zeros((n, m))
	p = np.zeros((n, m))
	L = 1
	
	p[:, 0] = p1
	p[:, m - 1] = p2
	
	s[0] = 0.2
	s[:, 0] = 0.8
	#print(p)
	#print(s)
	for t in range(n - 1):
		a, b, c, h, tau = get_abch(s, t, L)
		p[t + 1] = progonka_count(a, b, c, p1, p2)
		s = s_iter(s, p, t, h, tau)
	print_mass(p[1:], "p mass:")
	print_onemas(s[-1], "s at the last moment")

def test():
	x = np.arange(0, m)
	y = np.exp(x)
	f = interpolate(x, y, fill_value="extrapolate")
	print(y)
	print(f(m))
	xnew = np.arange(0, m + 1, 1)
	ynew = f(xnew)
	plt.plot(x, y, 'o', xnew, ynew, '-')
	plt.show()

if __name__ == "__main__":
	main()

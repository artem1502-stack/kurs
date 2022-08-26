import numpy as np
from numba import njit

n = 10

nn1 = 1.
nn2 = 10.
ss1 = 0.25
ss2 = 0.2
m = 0.1

@njit
def B(s):
	return k1(s) / nn1 + k2(s) / nn2 

@njit
def k1(s):
	if s < ss1:
		return 0
	return ((s - ss1) / (1 - ss1)) ** 2

@njit
def k2(s):
	if s > ss2:
		return 0
	return (1 - (s / ss2)) ** 2

@njit
def phi(s):
	return (k1(s) / nn1) / (k1(s) / nn1 + k2(s) / nn2)


def s_iter(s, p, t, h, tau):

	for i in range(1, n - 1):
		aa1 = B((s[t][i] + s[t][i + 1]) / 2) * phi((s[t][i] + s[t][i + 1]) / 2)
		ab1 = (p[t + 1][i + 1] - p[t + 1][i]) / h
		aa2 = B((s[t][i - 1] + s[t][i]) / 2) * phi((s[t][i - 1] + s[t][i]) / 2)
		ab2 = (p[t + 1][i] - p[t + 1][i - 1]) / h
		
		print(f"aa1 = {aa1},\taa2 = {aa2}")
		s[t + 1][i] = s[t][i] + (tau / (m * h)) * (aa1 * ab1 - aa2 * ab2)
	s[t + 1][n - 1] = s[t][n - 1]
	return s

def progonka_count(a, b, c, p1, p2):
	y = np.zeros(n)
	alpha = np.zeros(n)
	beta = np.zeros(n)
	x = np.zeros(n)

	y[0] = b[0]
	alpha[0] = -c[0]/y[0]
	beta[0] = 0
	for i in range(1, n - 1):
		y[i] = b[i] + a[i] * alpha[i - 1]
		alpha[i] = -c[i] / y[i]
		beta[i] = -a[i] * beta[i - 1] / y[i]
	y[n - 1] = b[n - 1] + a[n - 1] * alpha[n - 2]
	beta[n - 1] = -a[n - 1] * beta[n - 2] / y[n - 1]

	x[n - 1] = p1
	for i in range(n - 2, -1, -1):
		x[i] = alpha[i] * x[i + 1] + beta[i]
	return x[::-1]

def get_abch(s, t, L):
	a, b, c = np.zeros(n), np.zeros(n), np.zeros(n)

	c[0] = B((s[t][0] + s[t][1]) / 2)
	b[0] = -2 * B(s[t][0])
	for i in range(1, n - 1):
		shm = (s[t][i - 1] + s[t][i]) / 2
		shp = (s[t][i] + s[t][i + 1]) / 2
		a[i] = B(shm)
		b[i] = -B(shm) - B(shp)
		c[i] = B(shp)
	a[n - 1] = B((s[t][n - 2] + s[t][n - 1]) / 2)
	b[n - 1] = -2 * B(s[t][n - 1])

	return a, b, c, (L / (n - 1))

def print_mass(m, s):
	print(s)
	for i in m:
		for j in i:
			print(f"{j:.2e}", end="\t")
		print("")


def main():
	p1 = 1
	p2 = 0
	s = np.zeros((n, n))
	p = np.zeros((n, n))
	L = 1
	
	p[:, 0] = p1
	p[:, n - 1] = p2
	
	s[:, 0] = 1
	print(p)
	print(s)
	t = 0
	a, b, c, h = get_abch(s, t, L)
	p[t] = progonka_count(a, b, c, p1, p2)
	s = s_iter(s, p, t, h, h)
	#for t in range(n - 1):
	#	a, b, c, h = get_abch(s, t, L)
	#	p[t] = progonka_count(a, b, c, p1, p2)
	#	s = s_iter(s, p, t, h, h)
	print_mass(p, "p mass:")
	print_mass(s, "s mass:")

if __name__ == "__main__":
	main()

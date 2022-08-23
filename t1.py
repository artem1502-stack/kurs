import numpy as np
from numbu import njit

n = 10

nn1 = 1.
nn2 = 10.
ss1 = 0.25
ss2 = 0.2

@njit
def B(s):
	return k1(s) / nn1 + k2(s) / nn2 


@njit
def k1(s):
	return s ** 2

@njit
def k2(s):
	return (1 - s) ** 2

@njit
def phi(s):
	(k1(s) / nn1) / (k1(s) / nn1 + k2(s) / nn2)


def progonka_count(a, b, c, h, p1, p2):
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

	x[n - 1] = p2
	for i in range(n - 2, -1, -1):
		x[i] = alpha[i] * x[i + 1] + beta[i]
	return x

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

def main():
	s = np.zeros((n, n))
	L = 1
	s[:][0] = 1
	t = 0
	a, b, c, h = get_abch(s, t, L)

	print(s)

if __name__ == "__main__":
	main()

import numpy as np
from numba import njit
from scipy.interpolate import interp1d as interpolate
import matplotlib.pyplot as plt

n = 100
m = 200
NNN = int(m / 15)

nn1 = 1.
nn2 = 10.
ss1 = 0.25
ss2 = 0.2
mc = 0.1
kk = 1

plt.rcParams['font.size'] = '8'

@njit
def B(s):
	return (k1(s) / nn1) + (k2(s) / nn2)
@njit
def W(s, dp):
	return (-kk * B(s) * dp)
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
def phi(s):
	return (k1(s) / nn1) / (k1(s) / nn1 + k2(s) / nn2)

def s_iter(s, p, t, h, tau):

	for i in range(1, m - 1):
		shp = (s[t][i] + s[t][i + 1]) / 2
		shm = (s[t][i - 1] + s[t][i]) / 2
		aa1 = B(shp) * phi(shp)
		ab1 = (p[t + 1][i + 1] - p[t + 1][i]) / h
		aa2 = B(shm) * phi(shm)
		ab2 = (p[t + 1][i] - p[t + 1][i - 1]) / h
		s[t + 1][i] = np.abs(s[t][i] + (tau / (mc * h)) * (aa1 * ab1 - aa2 * ab2))

	p_fun = interpolate(np.arange(m), p[t + 1], fill_value="extrapolate")
	p_lst = p_fun(m)
	ab1 = (p_lst - p[t + 1][i]) / h
	shp = s[t][i]
	shm = (s[t][i - 1] + s[t][i]) / 2
	aa1 = B(shp) * phi(shp)
	aa2 = B(shm) * phi(shm)
	ab2 = (p[t + 1][i] - p[t + 1][i - 1]) / h
	s[t + 1][i] = np.abs(s[t][i] + (tau / (mc * h)) * (aa1 * ab1 - aa2 * ab2))


	return s

@njit
def progonka_count(a, b, c, p1, p2, dd):
	y = np.zeros(m)
	alpha = np.zeros(m)
	beta = np.zeros(m)
	x = np.zeros(m)

	y[0] = b[0]
	if (np.abs(y[0]) <= 0):
		y[0] = -0.0001
	alpha[0] = -c[0] / y[0]
	beta[0] = 0
	for i in range(1, m - 1):
		y[i] = b[i] + a[i] * alpha[i - 1]
		if (np.abs(y[i]) <= 0):
			y[i] = -0.0001
		alpha[i] = -c[i] / y[i]
		beta[i] = -a[i] * beta[i - 1] / y[i]
	y[m - 1] = b[m - 1] + a[m - 1] * alpha[m - 2]
	if (np.abs(y[m - 1]) <= 0):
		y[m - 1] = -0.0001
	dd = p2 * dd
	beta[m - 1] = (dd-a[m - 1] * beta[m - 2]) / y[m - 1]

	x[m - 1] = p1
	for i in range(m - 2, -1, -1):
		x[i] = alpha[i] * x[i + 1] + beta[i]
	return x[::-1]
def get_abch(s, t, L):
	a, b, c = np.zeros(m), np.zeros(m), np.zeros(m)

	c[0] = B((s[t][0] + s[t][1]) / 2)
	b[0] = - B(s[t][0]) - B((s[t][0] + s[t][1]) / 2)
	for i in range(1, m - 1):
		shm = (s[t][i - 1] + s[t][i]) / 2
		shp = (s[t][i] + s[t][i + 1]) / 2
		a[i] = B(shm)
		b[i] = -B(shm) - B(shp)
		c[i] = B(shp)
	a[m - 1] = B((s[t][m - 2] + s[t][m - 1]) / 2)
	b[m - 1] = 0#- B(s[t][m - 1]) - B((s[t][m - 2] + s[t][m - 1]) / 2)
	dd = (B(s[t][m - 1]) + B((s[t][m - 2] + s[t][m - 1]) / 2))
	return a, b, c, (L / (n - 1)), dd

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

def pressure_theore(x):
	return x * (-9) + 10

def main():
	p1 = 10
	p2 = 1
	bt = 0
	s = np.zeros((n, m))
	p = np.zeros((n, m))
	L = 1

	p[:, 0] = p1
	p[:, m - 1] = p2

	s[0] = 0.1
	s[:, 0] = 0.3
	print(s)
	
	end = 1
	t = 0
	a, b, c, h, dd = get_abch(s, t, L)
	p[0] = progonka_count(a, b, c, p1, p2, dd)
	while (end > 1.e-5):
		a, b, c, h, dd = get_abch(s, t, L)
		p[t + 1] = progonka_count(a, b, c, p1, p2, dd)
		dp = (p[t + 1][1:] - p[t + 1][:-1]) / h
		###########
		warray = np.zeros(len(dp))
		for i, item in enumerate(dp):
			warray[i] = W(s[t][i], item)
		to_tau = max(np.abs(warray))
		tau =  2 * h  / to_tau
		###########
		bt += tau
		s = s_iter(s, p, t, h, tau)
		end = np.linalg.norm(s[t + 1] - s[t])
		print(f"\rtau = {tau:.2e}, end = {end}, n = {t}", end = "\r")
		t += 1
		if (t >= n - 2):
			print("\n Not enough space")
			break

	print("\n")
	#print_mass(p[1:t], "p mass:")
	#print_mass(s, "s at the last moment")

	print(f"n = {t}, time = {bt}")
	x = np.linspace(0, 1, m);
	#p = p * 9/10 + 1
	fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
	#fig2, (bx1, bx2, bx3) = plt.subplots(3, 1, sharex=True)
	
	fig.set_label("Graph")
	ax2.set_ylabel('Водонасыщенность')
	ax3.set_xlabel('Координаты')
	
	ax1.set_title("на 0-м шаге по времени")
	ax2.set_title("на середине по времени")
	ax3.set_title("на конце по времени")
	print(t)
	print(s[0:3])
	ax1.plot(x, s[0], 'r')
	
	ax2.plot(x, s[1], 'g')
	ax3.plot(x, s[2], 'b');
	fig.set_label("Graph")
	#bx2.set_ylabel('Давление')
	#bx3.set_xlabel('Координаты')
	
	#bx1.set_title("на 0-м шаге по времени")
	#bx2.set_title("на середине по времени")
	#bx3.set_title("на конце по времени")

	#bx1.plot(x, p[0], 'r')
	#bx2.plot(x, p[t // 2], 'g')
	#bx3.plot(x, p[t - 1], 'b')
	
	#ax1.set_yscale('log')
	#ax2.set_yscale('log')

	ax1.set_ylim([0, 1])
	ax2.set_ylim([0, 1])
	ax3.set_ylim([0, 1])
	plt.show()


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

def test_2():
	x = np.linspace(0, 1, 100)
	y = np.vectorize(phi)(x)
	z = np.vectorize(B)(x)
	plt.plot(x, y, "r", x, z, "g")
	plt.show()

if __name__ == "__main__":
	main()

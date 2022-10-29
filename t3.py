import numpy as np
from numba import njit
from scipy.interpolate import interp1d as interpolate
import matplotlib.pyplot as plt

n = 1000
m = 50

nn1 = 1.
nn2 = 10.
ss1 = 0.2
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

@njit
def pressure_theore(x):
	return x * (-9) + 10

@njit
def s_iter(s, p, t, h, tau):

	for i in range(1, m - 1):
		shp = (s[t][i] + s[t][i + 1]) / 2
		shm = (s[t][i - 1] + s[t][i]) / 2
		aa1 = B(shp) * phi(shp)
		ab1 = (p[i + 1] - p[i]) / h
		aa2 = B(shm) * phi(shm)
		ab2 = (p[i] - p[i - 1]) / h
		s[t + 1][i] = np.abs(s[t][i] + (tau / (mc * h)) * (aa1 * ab1 - aa2 * ab2))

	p_lst = pressure_theore(1 + h)
	ab1 = (p_lst - p[m - 1]) / h
	shp = s[t][m - 1]
	shm = (s[t][m - 2] + s[t][m - 1]) / 2
	aa1 = B(shp) * phi(shp)
	aa2 = B(shm) * phi(shm)
	ab2 = (p[m - 1] - p[m - 2]) / h
	s[t + 1][m - 1] = np.abs(s[t][m - 1] + (tau / (mc * h)) * (aa1 * ab1 - aa2 * ab2))

	return s

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
	bt = 0
	s = np.zeros((n, m))
	L = 1

	p = np.vectorize(pressure_theore)(np.linspace(0, 1, m))
	s[0] = 0.2
	s[:, 0] = 0.3
	print(s)
	
	end = 1
	t = 0
	h = 1 / (m - 1)
	
	###########
	dp = (p[1:] - p[:-1]) / h
	warray = np.zeros(len(dp))
	for i, item in enumerate(dp):
		warray[i] = W(s[t][i], item)
	w_max = max(np.abs(warray))
	print(f"Wmax = {w_max}")
	tau =  0.5 * h  / w_max
	if (tau >= 0) and (tau <= 0):
		print("Use huyna")
		return
	###########

	while (end > 1.e-5):
		bt += tau
		s = s_iter(s, p, t, h, tau)
		end = np.linalg.norm(s[t + 1] - s[t])
		print(f"\rtau = {tau:.2e},\tend = {end:.2e},\tn = {t}\t\tCompleted: {100 * t / (n - 2):.2f}%",end = "\r")
		t += 1
		if (t >= n - 2):
			print("\n Not enough space")
			break
	print("\n")

	print(f"n = {t}, time = {bt}")
	x = np.linspace(0, 1, m);
	fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)

	fig.set_label("Graph")
	ax2.set_ylabel('Водонасыщенность')
	ax3.set_xlabel('Координаты')

	ax1.set_title("на 0-м шаге по времени")
	ax2.set_title("на середине по времени")
	ax3.set_title("на конце по времени")
	print(t)
	#print(s[::t // 2])
	ax1.plot(x, s[0], 'r', x, s[(t // 2) // 3], 'g', s[2 * (t // 2) // 3], 'b')
	ax2.plot(x, s[t // 2], 'r', x, s[4 * (t // 2) // 3], 'g', s[5 * (t // 2) // 3], 'b')
	ax3.plot(x, s[t], 'r')
	fig.set_label("Graph")

	ax1.set_ylim([0, 1])
	ax2.set_ylim([0, 1])
	ax3.set_ylim([0, 1])

	plt.show()

def test():
	x = np.linspace(0, 1, 1000)
	y = np.vectorize(phi)(x)
	plt.plot(x,y, "r")
	plt.show()

if __name__ == "__main__":
	test()

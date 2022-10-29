import numpy as np
from numba import njit
from scipy.interpolate import interp1d as interpolate
import matplotlib.pyplot as plt

n = 1000
m = 500

nn1 = 1.
nn2 = 5
ss1 = 0.1
ss2 = 0.1
mc = 0.1
kk = 1

plt.rcParams['font.size'] = '8'

@njit
def normalize(s):
	for i in range(1, n):
		s[i] /= np.linalg.norm(s[i])
		ds = 1
		rs = s[i][0]
		for j in range(1, m):
			ds = s[i][j] - s[i][j - 1]
			if np.abs(ds) < 1.e-10:
				rs = j
			break
		s[i][:j] += s[0][0] - s[i][0]
		s[i][j:] += s[0][1] - s[i][j]
	return s

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
def bb(s, dp):
	return -20 * dp * kk * s / nn1

@njit
def s_iter(s, dp, t, h, tau):
	ww = bb(s[t], dp=dp)
	s[t + 1][1:] = s[t][1:] - ww[1:] * (tau / h) * (s[t][1:] - s[t][:-1])
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
	global n
	bt = 0
	L = 1

	p = np.vectorize(pressure_theore)(np.linspace(0, 1, m))
	
	end = 1
	t = 0
	h = 1 / (m - 1)

	###########
	dp = (p[1] - p[0]) / h
	warray = np.zeros(m)
	pos_s = np.linspace(0, 1, m)

	for i, item in enumerate(pos_s):
		warray[i] = W(item, dp)
	w_max = max(np.abs(warray))
	print(f"Wmax = {w_max}")
	tau =  0.1 * h  / w_max
	###########
	t_progn = 1 / w_max
	n_prob = t_progn / tau
	print(f"n_prob = {n_prob}")
	n = int(n_prob) * 10
	s = np.zeros((n, m))
	s[0] = 0.2
	s[:, 0] = 0.3
	

	while (t < n):
		bt += tau
		s = s_iter(s, p, t, h, tau)
		print(f"\rtau = {tau:.2e},\tend = {end:.2e},\tn = {t}\t\tCompleted: {100 * t / (n - 2):.2f}%",end = "\r")
		t += 1
	print("\n")

	#s = normalize(s)
	print(f"n = {t}, time = {bt}")
	x = np.linspace(0, 1, m);
	fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)

	fig.set_label("Graph")
	ax2.set_ylabel('Водонасыщенность')
	ax3.set_xlabel('Координаты')

	#ax1.set_title("на 0-м шаге по времени")
	#ax2.set_title("на середине по времени")
	#ax3.set_title("на конце по времени")
	print(t)
	#print(s[::t // 2])
	#ax1.plot(x, s[0], 'r', x, s[(t // 2) // 3], 'g', s[2 * (t // 2) // 3], 'b')
	#ax2.plot(x, s[t // 2], 'r', x, s[4 * (t // 2) // 3], 'g', s[5 * (t // 2) // 3], 'b')
	#ax3.plot(x, s[t], 'r')
	
	


	ax1.plot(x, s[0], "r")
	ax2.plot(x, s[t // 3], "g")
	ax3.plot(x, s[2 * t // 3], "b")
	ax4.plot(x, s[t - 1], "black")
	fig.set_label("Graph")

	ax1.set_ylim([0, 1])
	ax2.set_ylim([0, 1])
	ax3.set_ylim([0, 1])
	ax4.set_ylim([0, 1])
	
	plt.show()

def test():
	x = np.linspace(0, 1, 1000)
	y = np.vectorize(pressure_theore)(x)
	plt.plot(x,y, "r")
	plt.show()

if __name__ == "__main__":
	main()

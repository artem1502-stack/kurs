from functions import *
import sys

def s_iter(s, dp, t, h, tau):

	ww = bb(s[t], dp=dp)
	s[t + 1][1:] = s[t][1:] - ww[1:] * (tau / h) * (s[t][1:] - s[t][:-1])
	return s

@njit
def progonka_count(a, b, c, p1, p2, dd = 0):
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
	beta[m - 1] = (dd-a[m - 1] * beta[m - 2]) / y[m - 1]

	x[m - 1] = p1
	for i in range(m - 2, -1, -1):
		x[i] = alpha[i] * x[i + 1] + beta[i]
	return x[::-1]

def get_abch(s):
	a, b, c = np.zeros(m), np.zeros(m), np.zeros(m)

	c[0] = B((s[0] + s[1]) / 2)
	b[0] = -2 * B(s[0])
	for i in range(1, m - 1):
		shm = (s[i - 1] + s[i]) / 2
		shp = (s[i] + s[i + 1]) / 2
		a[i] = B(shm)
		b[i] = -B(shm) - B(shp)
		c[i] = B(shp)
	a[m - 1] = B((s[m - 2] + s[m - 1]) / 2)
	b[m - 1] = -2 * B(s[m - 1])

	return a, b, c

def main():
	global n
	s = np.zeros((n * 10, m))
	p = np.zeros((n * 10, m))

	s[0] = s2
	s[:, 0] = s1

	h = L / (m - 1)

	a, b, c = get_abch(s[0])
	p[1] = progonka_count(a, b, c, p1, p2)
	
	dp = np.average((p[1][1:] - p[1][:-1]) / h)

	s_pos = np.linspace(0, L, m)
	w_water = bb(s_pos, dp=dp)
	w_max = max(w_water)
	tau =  0.5 * h  / w_max
	
	s = s_iter(s, dp, 0, h, tau)
	
	n_m = int((1 / w_max) / tau) * 5
	print(n_m)
	print(s)
	counter = 0
	for t in range(1, n - 1):
		s = s_iter(s, dp, t, h, tau)
		a, b, c = get_abch(s[t])
		p[t + 1] = progonka_count(a, b, c, p1, p2)
		dp = np.average((p[t + 1][1:] - p[t + 1][:-1]) / h)
		if t > n_m:
			break
	print("Count complete")
	#print(s)
	print("\n")
	return s[:min(n, n_m)]


def graph(s):
	fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)

	fig.set_label("Graph")
	ax2.set_ylabel('Водонасыщенность')
	ax3.set_xlabel('Координаты')

	x = np.linspace(0, 1, m)

	n_loc = len(s)

	ax1.plot(x, s[0], "r")
	print(s[n_loc // 3])
	ax2.plot(x, s[n_loc // 3], "g")
	ax3.plot(x, s[2 * n_loc // 3], "b")
	ax4.plot(x, s[n_loc - 1], "black")
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
	graph(main())

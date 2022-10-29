import numpy as np
import matplotlib.pyplot as plt

h = 1.e-5

def f(x):
	return ((x + 2) ** 2) * (x + 8) - 7

def df(x):
	return (f(x + h) - f(x)) / h

def pf(x):
	return 3 * (x**2 + 8 * x + 12)

xx = np.linspace(-13, 0, 2000)
y = np.vectorize(df)(xx)
yy = np.vectorize(pf)(xx)

plt.plot(xx, y, "r")
plt.plot(xx, yy, "b")
plt.show()

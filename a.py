from scipy import integrate
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-2, 2, num=20)
y = x

print(x)
print(y)

y_int = integrate.cumtrapz(y, x, initial=0)
plt.plot(x, y_int, 'ro', x, y[0] + 0.5 * x**2, 'b-')
plt.show()
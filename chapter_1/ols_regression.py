import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 2 * x ** 2 - x ** 3 / 3

x = np.linspace(-2, 4, 25)

y = f(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'ro')
plt.grid(True)
plt.show()

# Calculation of optimal beta
beta = np.cov(x, y, ddof=0)[0, 1] / np.var(x)

# Calculation of optimal alpha
alpha = y.mean() - beta * x.mean()

# Calculation of estimated output values
y_ = alpha + beta * x

# Calculation of the MSE given the approximation
MSE = ((y - y_) ** 2).mean()

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'ro', label='sample data')
plt.plot(x, y_, lw=3.0, label='linear regression')
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'ro', label='sample data')
for deg in [1, 2, 3]:
    reg = np.polyfit(x, y, deg=deg)
    y_ = np.polyval(reg, x)
    MSE = ((y - y_) ** 2).mean()
    print(f'deg={deg} | MSE={MSE:.5f}')
    plt.plot(x, np.polyval(reg, x), label=f'deg={deg}')
plt.legend();
plt.show()
# Arbitrage pricing
import numpy as np
from scipy.optimize import minimize

import matplotlib.pyplot as plt
import matplotlib as mpl

# Price of stocks and bound today
# Stock = Company stock value
# Bound = Like a Loan. Debit of a company
S0 = 10
B0 = 10

# The uncertain payoff of the stock and bond tomorrow
S1 = np.array((20, 5))
B1 = np.array((11, 11))

# The market price vector
M0 = np.array((S0, B0))

# The market payoff matrix
M1 = np.array((S1, B1)).T

# strike price of the option (the difference expected, the return)
K = 14.5

# The uncertain payoff of the option
C1 = np.maximum(S1 - K, 0)

# The replication portfolio ϕ
phi = np.linalg.solve(M1, C1)

# A check whether its payoff is the same as the option’s payoff
np.allclose(C1, np.dot(M1, phi))

# The price of the replication portfolio is the arbitrage-free price of the option
C0 = np.dot(M0, phi)

print(C0)

#---------------
# Expected Utility Theory
def u(x):
    # The risk-averse Bernoulli utility function
    return np.sqrt(x)

def EUT(x):
    # The expected utility function
    return np.dot(P, u(x))

# Two portfolios with different weights
phi_A = np.array((0.75, 0.25))
phi_D = np.array((0.25, 0.75))

# Shows that the cost to set up each portfolio is the same
np.dot(M0, phi_A) == np.dot(M0, phi_D)

# The uncertain payoff of one portfolio
A1 = np.dot(M1, phi_A)

# The uncertain payoff of another portfolio
D1 = np.dot(M1, phi_D)

# The probability measure
P = np.array((0.5, 0.5))

# The utility values for the two uncertain payoffs
print(EUT(A1))
print(EUT(D1))

#-----------------------------
# The fixed budget of the agent
w = 10

# The budget constraint for use with minimize
cons = {'type':'eq', 'fun': lambda phi: np.dot(M0, phi) - w}

def EUT_(phi):
    # The expected utility function defined over portfolios
    x = np.dot(M1, phi)
    return EUT(x)

# Minimizing -EUT_(phi) maximizes EUT_(phi)
opt = minimize(lambda phi: - EUT_(phi), 
        # The initial guess for the optimization
        x0=phi_A, 
        # The budget constraint applied
        constraints=cons)

print("OPT", opt)

# The optimal (highest) expected utility given the budget of w = 10
print(EUT_(opt['x']))

# ---------------------------------------
# Numerical Example

# Vector of the risky asset
rS = S1 / S0 -1

rB = B1 / B0 -1 

def mu(rX):
    return np.dot(P, rX)

print("RS: ", mu(rS))
print("RB: ", mu(rB))

# Return matrix for the traded assets
rM = M1 / M0 - 1

# Expected return vector
print("RM: ", mu(rM))

#---------------------------------------
def var(rX):
    # The variance function
    return ((rX - mu(rX)) ** 2).mean()

print("RS: ", var(rS))

print("RB: ", var(rB))

def sigma(rX):
    # The volatility function
    return np.sqrt(var(rX))

print("Sigma RS: ", sigma(rS))

print("Sigma RB: ", sigma(rB))

# The covariance matrix
np.cov(rM.T, aweights=P, ddof=0)

# portfolio expected return, portfolio expected variance, and portfolio expected volatility
phi = np.array((0.5, 0.5))

def mu_phi(phi):
    # The portfolio expected return
    return np.dot(phi, mu(rM))

print(mu_phi(phi))

def var_phi(phi):
    cv = np.cov(rM.T, aweights=P, ddof=0)
    # The portfolio expected variance
    return np.dot(phi, np.dot(cv, phi))

print(var_phi(phi))

def sigma_phi(phi):
    # The portfolio expected volatility
    return var_phi(phi) ** 0.5

print(sigma_phi(phi))

#----------------------------------
# Investment opportunity set


plt.style.use('seaborn')
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.family'] = 'serif'

# Random portfolio compositions, normalized to 1
phi_mcs = np.random.random((2, 200))
phi_mcs = (phi_mcs / phi_mcs.sum(axis=0)).T

# Expected portfolio volatility and return for the random compositions
mcs = np.array([(sigma_phi(phi), mu_phi(phi)) for phi in phi_mcs])

plt.figure(figsize=(10, 6))
plt.plot(mcs[:, 0], mcs[:, 1], 'ro')
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.show()

# Monte Carlo simulation
# New probability measure for three states
P = np.ones(3) / 3

S1 = np.array((20, 10, 5))

T0 = 10
T1 = np.array((1, 12, 13))

M0 = np.array((S0, T0))
M1 = np.array((S1, T1)).T

rM = M1 / M0 - 1

mcs = np.array([(sigma_phi(phi), mu_phi(phi)) for phi in phi_mcs])

plt.figure(figsize=(10, 6))
plt.plot(mcs[:, 0], mcs[:, 1], 'ro')
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.show()

# Minimum volatility and maximum Sharpe ratio

cons = {'type': 'eq', 'fun': lambda phi: np.sum(phi) - 1}
bnds = ((0, 1), (0, 1))

min_var = minimize(sigma_phi, (0.5, 0.5), constraints=cons, bounds=bnds)

def sharpe(phi):
    return mu_phi(phi) / sigma_phi(phi)

max_sharpe = minimize(lambda phi: -sharpe(phi), (0.5, 0.5), constraints=cons, bounds=bnds)

plt.figure(figsize=(10, 6))
plt.plot(mcs[:, 0], mcs[:, 1], 'ro', ms=5)
plt.plot(sigma_phi(min_var['x']), mu_phi(min_var['x']), '^', ms=12.5, label='minimum volatility')
plt.plot(sigma_phi(max_sharpe['x']), mu_phi(max_sharpe['x']), 'v', ms=12.5, label='maximum Sharpe ratio')
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.legend()
plt.show()
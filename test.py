import numpy as np
from scipy.optimize import minimize

def f(x0, x1, alpha=0.0):
    x0 = x0.reshape(x1.shape)
    return np.sum(np.sum((x1-x0)**2.0)) + alpha*np.sum(np.sum((x0[1:,:]-x0[:-1,:])**2.0)+ (x0[0,:]-x0[-1,:])**2.0)

x0 = np.array([[0,0],[1,0],[2,0]])
x1 = np.array([[0,1],[1,1],[2,1]])

res = minimize(f, x0, method='BFGS', tol=1e-8, args=x1)
print(res)
print(res.x)

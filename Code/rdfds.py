import numpy as np
from functions import noisy_paired_values
from generator import sphere_point_generator
from newton import backtracking_line_search, newton
import time

def approximate_gradient(func, x, e, t, m):
    (f_te, f) = func([x+t*e, x], m)
    gradient = ((f_te - f)/t).mean() * e
    return gradient

def V_function(alpha, eg, z, x, order=0):
    z = np.asmatrix(z)
    x = np.asmatrix(x)
    n = x.shape[0]
 
    if order == 0:
        value = alpha * n * eg.T * (x-z) + 0.5 * (x-z).T * (x-z)
        return value
    if order == 1:
        value = alpha * n * eg.T * (x-z) + 0.5 * (x-z).T * (x-z)
        gradient = alpha * n * eg + x-z
        return (value, gradient)
    if order == 2:
        value = 0.5 * (z-x).T * (z-x)
        gradient = alpha * n * eg + x-z
        hessian = np.identity(x.shape[0])
        return (value, gradient, hessian)



def ardfds(func, L, m, t, initial_x, maximum_iterations=1000, ardfds_eps=1e-5, newton_eps=1e-5):
    '''
    m:              batch size when computing gradient.
    t:              smoothing parameter when computing gradient.
    '''
    
    if ardfds_eps <= 0:
        raise ValueError("Epsilon must be positive")
    x = np.matrix(initial_x)
    y = x.copy()
    z = x.copy()
    n = x.shape[0]
    # initialization
    values = []
    runtimes = []
    xs = []
    start_time = time.time()
    iterations = 0

    sphere_point_G = sphere_point_generator(n)

    for k in range(maximum_iterations):
        alpha = (k+2) / (96 * n * n * L)
        tau = 2 / (k + 2)
        e = sphere_point_G(1)
        value = func(x,1)
        print(value)
        gradient = approximate_gradient(func, x, e, t, m)
        x = tau * z + (1-tau) * y
        y = x - 1/(2* L) * gradient
        newton_f = lambda x, order: V_function(alpha, gradient, z, x, order)
        z, newton_values, runtimes, xs = newton( newton_f, x, newton_eps, 100, backtracking_line_search)
        values.append(y)
    return y[-100:].mean()







import numpy as np
from functions import noisy_paired_values
from generator import sphere_point_generator
from newton import backtracking_line_search, newton
import time

import pdb

def approximate_gradient(func, x, e, mu, m):
    f_te = func(x+mu*e, m)
    f = func(x, m)
    gradient = ((f_te - f)/mu).mean() * e
    # print(gradient)
    # pdb.set_trace()
    return gradient


def stars(func, L, m, mu, initial_x, maximum_iterations=1000, stars_eps=1e-5, noise_mode=0):
    '''
    m:              batch size when computing gradient.
    mu:              smoothing parameter when computing gradient.
    '''
    
    if stars_eps <= 0:
        raise ValueError("Epsilon must be positive")
    x = np.matrix(initial_x)
    n = x.shape[0]

    # initialization
    values = []
    runtimes = []
    xs = []
    start_time = time.time()
    iterations = 0

    sphere_point_G = sphere_point_generator(n)
    h = 1./(4*L*(n+4))
    # print(1./(4*L*(n+4)))

    if noise_mode == 0: #Additive noise
        for k in range(maximum_iterations):
            e = sphere_point_G(1)
            value = func(x,1)[0,0]
            values.append(value)
            gradient = approximate_gradient(func, x, e, mu, m)
            x = x - h*gradient
            # print(x)
        return x

    if noise_mode == 1:
        for k in range(maximum_iterations):
            e = sphere_point_G(1)
            value = func(x,1)[0,0]
            values.append(value)
            f = func(x, m)
            fx = f.mean()
            mu1 = mu*np.power(np.abs(fx),0.5)
            gradient = approximate_gradient(func, x, e, mu1, m)
            x = x - h*gradient
        return x

        











# def ardfds(func, L, m, t, initial_x, maximum_iterations=1000, ardfds_eps=1e-5, newton_eps=1e-5):
#     '''
#     m:              batch size when computing gradient.
#     t:              smoothing parameter when computing gradient.
#     '''
    
#     if ardfds_eps <= 0:
#         raise ValueError("Epsilon must be positive")
#     x = np.matrix(initial_x)
#     y = x.copy()
#     z = x.copy()
#     n = x.shape[0]
#     # initialization
#     values = []
#     runtimes = []
#     xs = []
#     start_time = time.time()
#     iterations = 0

#     sphere_point_G = sphere_point_generator(n)

#     for k in range(maximum_iterations):
#         alpha = (k+2) / (96 * n * n * L)
#         tau = 2 / (k + 2)
#         e = sphere_point_G(1)
#         value = func(x,1)[0,0]
#         gradient = approximate_gradient(func, x, e, t, m)
#         x = tau * z + (1-tau) * y
#         y = x - 1/(2* L) * gradient
#         newton_f = lambda x, order: V_function(alpha, gradient, z, x, order)
#         z, newton_values, runtimes, _ = newton( newton_f, x, newton_eps, 100, backtracking_line_search)
#         values.append(value)
#         xs.append(y)
#     return y








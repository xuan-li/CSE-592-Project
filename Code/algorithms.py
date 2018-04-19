import numpy as np
from functions import noisy_paired_values
from generator import sphere_point_generator
from newton import backtracking_line_search, newton
import time


def approximate_gradient(func, x, direction, t, m, mode = 2):
    '''
    direction: direction vector
    t: smooth parameter
    m: batch size
    mode: 2 - two-point feedback
          1 - one-point feedback
    '''
    if mode == 2:
        (f_te, f) = func([x+t*direction, x], m)
    else:
        f_te = func(x+t*direction, m)
        f = func(x, m)
    gradient = ((f_te - f)/t).mean() * direction
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
        value = alpha * n * eg.T * (x-z) + 0.5 * (x-z).T * (x-z)
        gradient = alpha * n * eg + x-z
        hessian = np.identity(x.shape[0])
        return (value, gradient, hessian)
        


def ardfds(func, initial_x,  L, m, t, maximum_iterations=1000, direction_generator = None, newton_eps=1e-5):
    '''
    m:              batch size when computing gradient.
    t:              smoothing parameter when computing gradient.
    '''
    x = np.matrix(initial_x)
    y = x.copy()
    z = x.copy()
    n = x.shape[0]
    # initialization
    xs = []
    xs.append(x)

    if not direction_generator:
        direction_generator = sphere_point_generator(n)

    for k in range(maximum_iterations):
        alpha = (k+2) / (96 * n * n * L)
        tau = 2 / (k + 2)
        e = direction_generator(1)
        gradient = approximate_gradient(func, x, e, t, m, 2)
        x = tau * z + (1-tau) * y
        y = x - 1/(2* L) * gradient
        newton_f = lambda x, order: V_function(alpha, gradient, z, x, order)
        z, newton_values, runtimes, _ = newton( newton_f, x, newton_eps, 100, backtracking_line_search)
        xs.append(y)
    return y, xs


def rdfds(func, initial_x, L, m, t, maximum_iterations=1000, direction_generator = None, newton_eps=1e-5):
    '''
    m:              batch size when computing gradient.
    t:              smoothing parameter when computing gradient.
    '''
    
    x = np.matrix(initial_x)
    n = x.shape[0]
    # initialization
    xs = []
    xs.append(x)

    if not direction_generator:
        direction_generator = sphere_point_generator(n)

    for k in range(maximum_iterations):
        alpha = 1 / (48 * n * L)
        e = direction_generator(1)
        gradient = approximate_gradient(func, x, e, t, m, 2)
        newton_f = lambda z, order: V_function(alpha, gradient, x, z, order)
        x, newton_values, runtimes, _ = newton( newton_f, x, newton_eps, 100, backtracking_line_search)
        xs.append(x)
    return x, xs


def stars(func, initial_x, L, m, mu, maximum_iterations=1000, noise_mode=0, direction_generator = None):
    '''
    m:              batch size when computing gradient.
    mu:              smoothing parameter when computing gradient.
    '''
    
    x = np.matrix(initial_x)
    n = x.shape[0]

    # initialization
    xs = []
    xs.append(x)

    if not direction_generator:
        direction_generator = sphere_point_generator(n)
    h = 1./(4*L*(n+4))
    # print(1./(4*L*(n+4)))

    if noise_mode == 0: #Additive noise
        for k in range(maximum_iterations):
            e = direction_generator(1)
            gradient = approximate_gradient(func, x, e, mu, m, 1)
            x = x - h*gradient
            xs.append(x)
            # print(x)
        return x,xs

    if noise_mode == 1:
        for k in range(maximum_iterations):
            e = direction_generator(1)
            f = func(x, m)
            fx = f.mean()
            mu1 = mu*np.power(np.abs(fx),0.5)
            gradient = approximate_gradient(func, x, e, mu1, m, 1)
            x = x - h*gradient
            xs.append(x)
        return x,xs

def rg(func, initial_x, L, m, mu, maximum_iterations=1000, noise_mode=0, direction_generator = None):
    '''
    m:              batch size when computing gradient.
    mu:              smoothing parameter when computing gradient.
    '''
    
    x = np.matrix(initial_x)
    n = x.shape[0]

    # initialization
    xs = []
    xs.append(x)

    if not direction_generator:
        direction_generator = sphere_point_generator(n)
    h = 1./(4*L*(n+4))
    # print(1./(4*L*(n+4)))

    if noise_mode == 0: #Additive noise
        for k in range(maximum_iterations):
            e = direction_generator(1)
            gradient = approximate_gradient(func, x, e, mu, m, 2)
            x = x - h*gradient
            xs.append(x)
            # print(x)
        return x,xs

    if noise_mode == 1:
        for k in range(maximum_iterations):
            e = direction_generator(1)
            f = func(x, m)
            fx = f.mean()
            mu1 = mu*np.power(np.abs(fx),0.5)
            gradient = approximate_gradient(func, x, e, mu1, m, 2)
            x = x - h*gradient
            xs.append(x)
        return x,xs
        





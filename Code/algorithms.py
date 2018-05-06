import numpy as np
from functions import noisy_paired_values
from generator import sphere_point_generator
from generator import gaussian_point_generator
from newton import backtracking_line_search, newton
import time


def approximate_gradient(func, x, direction_generator, t, m=1):
    '''
    direction: direction vector
    t: smooth parameter
    m: batch size
    mode: 2 - two-point feedback
          1 - one-point feedback
    '''
    direction = direction_generator(1)
    (f_te, f) = func([x+t*direction, x], m)
    gradient = ((f_te - f)/t).mean() * direction
    return gradient


def approximate_gradient_two_steps(func, x, current_state, last_state, direction_generator, t, m=1):
    '''
    direction: direction vector
    t: smooth parameter
    m: batch size
    mode: 2 - two-point feedback
          1 - one-point feedback
    '''
    direction = direction_generator(1)
    f_te = func(x+t*direction, m, explicit_noise = current_state)
    f = func(x, m, explicit_noise = last_state)
    gradient = ((f_te - f)/t).mean() * direction
    return gradient


def approximate_gradient_multi_direction(func, x, direction_generator, t, m):
    '''
    direction: direction vector
    t: smooth parameter
    m: batch size
    mode: 2 - two-point feedback
          1 - one-point feedback
    '''
    directions = direction_generator(m)
    gradient = 0
    for i in range(m):

        direction = directions[:,i]
        (f_te, f) = func([x+t*direction, x], 1)
        gradient += (f_te - f).mean()/t * direction
    return gradient / m



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
        
def mirror_descent(x, gradient, step_size):
    V = x - step_size * gradient
    return V

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
    xs.append(x.copy())
    runtimes = []
    start_time = time.time()
    runtimes.append(time.time() - start_time)
    if not direction_generator:
        direction_generator = sphere_point_generator(n)

    for k in range(maximum_iterations):
        alpha = (k+2) / (96 * n * n * L)
        tau = 2 / (k + 2)
        gradient = approximate_gradient(func, x, direction_generator, t, m)
        x = tau * z + (1-tau) * y
        y = x - 1/(2* L) * gradient
        z = mirror_descent(z, gradient, alpha * n)
        #newton_f = lambda x, order: V_function(alpha, gradient, z, x, order)
        #z, newton_values, _, _ = newton( newton_f, x, newton_eps, 100, backtracking_line_search)
        xs.append(y.copy())
        runtimes.append(time.time() - start_time)
    return y, xs ,runtimes,"ARDFDS"


def rdfds(func, initial_x, L, m, t, maximum_iterations=1000, direction_generator = None, newton_eps=1e-5):
    '''
    m:              batch size when computing gradient.
    t:              smoothing parameter when computing gradient.
    '''
    
    x = np.matrix(initial_x)
    n = x.shape[0]
    # initialization
    xs = []
    xs.append(x.copy())

    runtimes = []
    start_time = time.time()
    runtimes.append(time.time() - start_time)

    if not direction_generator:
        direction_generator = sphere_point_generator(n)

    for k in range(maximum_iterations):
        alpha = 1 / (48 * n * L)
        gradient = approximate_gradient(func, x, direction_generator, t, m)
        #newton_f = lambda z, order: V_function(alpha, gradient, x, z, order)
        #x, newton_values, _, _ = newton( newton_f, x, newton_eps, 100, backtracking_line_search)
        x = mirror_descent(x, gradient, alpha * n)
        xs.append(x.copy())
        runtimes.append(time.time() - start_time)
    return x,xs,runtimes,"RDFDS"


def rg(func, initial_x, L, m, mu, maximum_iterations=1000, direction_generator = None):
    '''
    m:              batch size when computing gradient.
    mu:              smoothing parameter when computing gradient.
    '''
    
    x = np.matrix(initial_x)
    n = x.shape[0]

    # initialization
    xs = []
    xs.append(x.copy())
    runtimes = []
    start_time = time.time()
    runtimes.append(time.time() - start_time)

    if not direction_generator:
        direction_generator = gaussian_point_generator(n)
    h = 1./(4*L*(n+4))

    for k in range(maximum_iterations):
        gradient = approximate_gradient(func, x, direction_generator, mu, m)
        x = x - h*gradient
        xs.append(x.copy())
        runtimes.append(time.time() - start_time)
    return x,xs,runtimes,"RG"


def stars(func, initial_x, L, m, mu, noise_generator, maximum_iterations=1000, direction_generator = None):
    '''
    m:              batch size when computing gradient.
    mu:              smoothing parameter when computing gradient.
    '''
    
    x = np.matrix(initial_x)
    n = x.shape[0]

    # initialization
    xs = []
    xs.append(x.copy())
    runtimes = []
    start_time = time.time()
    runtimes.append(time.time() - start_time)

    if not direction_generator:
        direction_generator = gaussian_point_generator(n)
    h = 1./(4*L*(n+4))

    last_random_state = noise_generator(1)
    for k in range(maximum_iterations):
        current_random_state = noise_generator(1)
        gradient = approximate_gradient_two_steps(func, x, last_random_state, current_random_state, direction_generator, mu, m)
        x = x - h*gradient
        xs.append(x.copy())
        runtimes.append(time.time() - start_time)
        last_random_state = current_random_state

    return x,xs,runtimes,"STARS"
        

def rsgf(func, initial_x, L, m, mu, maximum_iterations=1000, initial_stepsize = 1, direction_generator = None, two_phase = False):
    '''
    m:              batch size when computing gradient.
    mu:              smoothing parameter when computing gradient.
    '''
    
    x = np.matrix(initial_x)
    n = x.shape[0]

    # initialization
    xs = []
    xs.append(x.copy())
    runtimes = []
    start_time = time.time()
    runtimes.append(time.time() - start_time)

    if not direction_generator:
        direction_generator = gaussian_point_generator(n)

    h = 1 / np.sqrt(n+4) * min(1/(4 * L * np.sqrt(n+4)), initial_stepsize / np.sqrt(maximum_iterations))
    
    for k in range(maximum_iterations):
        gradient = approximate_gradient(func, x, direction_generator, mu, 1)
        x = x - h*gradient
        xs.append(x.copy())
        runtimes.append(time.time() - start_time)


    return x,xs,runtimes,"RSGF"





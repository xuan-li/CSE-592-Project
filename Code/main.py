from generator import gaussian_noise_generator
from generator import sphere_point_generator
from generator import gaussian_point_generator
from functions import noisy_function
from functions import sphere_function
from functions import nesterov_function
from functions import quadratic
import algorithms as alg
from functions import compute_L
import numpy as np
from utilities import plot_real_values
import matplotlib.pyplot as plt

sigma = 1e-4
noise_G = gaussian_noise_generator(sigma)
#func = lambda x, order=0: sphere_function(x, 2, order)
H = np.matrix('10 0; 0 40')
# the vector of linear coefficient of the quadratic function
b = np.matrix('0; 0')
#func = lambda x, order=0: quadratic(H, b,x, order)
func = lambda x, order=0: nesterov_function(x, order)
noisy_func = lambda x, n, explicit_noise=None: noisy_function(func, x, noise_G, n, explicit_noise = explicit_noise, noise_mode="multiply")
initial_x = np.matrix('100;100')
N = 1000
m = 100
t = 1e-2
t1 = 1e-4
n = initial_x.shape[0]
L = compute_L(func, n)
direction_G1 = sphere_point_generator(n)
direction_G2 = gaussian_point_generator(n)
#x0, xs0 = alg.ardfds(noisy_func, initial_x, L, m, t, N, feedback = 2, direction_generator = direction_G1)
'''
x1, xs1,time2 = alg.rsgf(
    noisy_func,
    initial_x,
    L,
    m,
    t1,
    N,
    initial_stepsize = 100,
    direction_generator=direction_G1,
    two_phase=False)
'''
#x2, xs2,time2 = alg.rg(noisy_func, initial_x, L, 1, t, N,  direction_generator = direction_G2)
#print("final x\n", x1)
x3, xs3,time2, _ = alg.stars(noisy_func, initial_x, L, 1, t, noise_generator=noise_G, maximum_iterations=N,  direction_generator = direction_G2)
print("final x\n", x3)
#print("final value\n", noisy_func(x,100000).mean())
fig = plt.figure()
#plot_real_values(func, xs1, fig, "1")
#plot_real_values(func, xs2, fig, "2")
plot_real_values(func, xs3, fig, "3")
plt.legend()
plt.show()

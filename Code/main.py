from generator import gaussian_noise_generator
from generator import sphere_point_generator
from generator import gaussian_point_generator
from functions import noisy_function
from functions import sphere_function
from functions import quadratic
import algorithms as alg
from functions import compute_L
import numpy as np
from utilities import plot_real_values
import matplotlib.pyplot as plt


sigma = 1e-1
noise_G = gaussian_noise_generator(sigma)
#func = lambda x, order=0: sphere_function(x, 10, order)
H = np.matrix('10 2 1; 4 30 10; 5 8 20');
# the vector of linear coefficient of the quadratic function
b = np.matrix('3; 5; 9')
func = lambda x, order=0: quadratic( H, b, x, order )
noisy_func = lambda x, n: noisy_function(func, x, noise_G, n, noise_mode="add")
initial_x = np.matrix('5;5;5')
N = 1000
m = 1000
t = 1e-3
n = initial_x.shape[0]
L = compute_L(func, n)
direction_G1 = sphere_point_generator(n)
direction_G2 = gaussian_point_generator(n)
#x0, xs0 = alg.ardfds(noisy_func, initial_x, L, m, t, N, direction_generator = direction_G1)
x1, xs1 = alg.rg(noisy_func, initial_x, L, 1, t, N, noise_mode=0, direction_generator = direction_G2)
x2, xs2 = alg.stars(noisy_func, initial_x, L, 1, t, N, noise_mode=0, direction_generator = direction_G2)
#print("final x\n", x)
#print("final value\n", noisy_func(x,100000).mean())
fig = plt.figure()
#plot_real_values(func, xs0, fig, "RDFDS")
plot_real_values(func, xs1, fig, "RG")
plot_real_values(func, xs2, fig, "STARS")
plt.legend()
plt.show()
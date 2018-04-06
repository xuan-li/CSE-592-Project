from generator import gaussian_noise_generator
from functions import noisy_function
from functions import sphere_function
from rdfds import ardfds
import numpy as np

noise_G = gaussian_noise_generator(10)
func = lambda x: sphere_function(x, 10)
noisy_func = lambda x, n: noisy_function(func, x, noise_G, n, noise_mode="add")
initial_x = np.matrix('10;10;10;10;10;10;10;10;10;10') * 10
theta = 0.5 * (initial_x - 0.2).T * (initial_x - 0.2)
m= 1000
t = 10e-6
L = 1
ardfds_eps = 1e-4
newton_eps = 1e-6
maximum_iterations = 5000
x = ardfds(noisy_func, L, m, t, initial_x, maximum_iterations, ardfds_eps, newton_eps)
print("final", x)
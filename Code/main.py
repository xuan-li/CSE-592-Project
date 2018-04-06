from generator import gaussian_noise_generator
from functions import noisy_function
from functions import sphere_function
from rdfds import ardfds,rdfds
import numpy as np

noise_G = gaussian_noise_generator(1)
func = lambda x: sphere_function(x, 10)
noisy_func = lambda x, n: noisy_function(func, x, noise_G, n, noise_mode="multiply")
initial_x = np.matrix('10;10;10;10;10;10;10;10;10;10') 
theta = 0.5 * (initial_x - 0.2).T * (initial_x - 0.2)
m = 100
t = 10e-8
L = 1
newton_eps = 1e-10
maximum_iterations = 1000
x = ardfds(noisy_func, L, m, t, initial_x, maximum_iterations, newton_eps)
print("final x\n", x)
print("final value\n", noisy_func(x,1000).mean())
from generator import gaussian_noise_generator
from functions import noisy_function
from functions import sphere_function
from stars import stars
import numpy as np

sigma = 0.001
noise_G = gaussian_noise_generator(sigma)
func = lambda x: sphere_function(x, 10)
noisy_func = lambda x, n: noisy_function(func, x, noise_G, n, noise_mode="multiply")
# noisy_func = lambda x, n: noisy_function(func, x, noise_G, n, noise_mode="add")
initial_x = np.matrix('10;10;10;10;10;10;10;10;10;10') 
# theta = 0.5 * (initial_x - 0.2).T * (initial_x - 0.2)

m = 100
n = initial_x.shape[0]
L = 1

mu_add = (8*sigma*sigma*n)/(L*L*np.power(n+6,3))
mu_add = np.power(mu_add,0.25)
mu_mult = (16*sigma*sigma*n)/(L*L*(1+3*sigma*sigma)*np.power(n+6,3))
mu_mult = np.power(mu_mult,0.25)

# print(L)
# print(mu_add)

# newton_eps = 1e-10
maximum_iterations = 10000
# x = stars(noisy_func, L, m, mu_add, initial_x, maximum_iterations, stars_eps=1e-5, noise_mode=0)
x = stars(noisy_func, L, m, mu_mult, initial_x, maximum_iterations, stars_eps=1e-5,noise_mode=1)
print("final x\n", x)
print("final value\n", noisy_func(x,1000).mean())
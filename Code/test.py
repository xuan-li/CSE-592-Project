from generator import gaussian_noise_generator
from generator import sphere_point_generator
from generator import gaussian_point_generator
from functions import noisy_function
from functions import sphere_function
from functions import nesterov_function
from functions import quadratic
import algorithms as alg
from functions import compute_L
from functions import boyd_example_func
import numpy as np
from utilities import plot_real_values
import matplotlib.pyplot as plt
from functions import bell_curve

sigma = 1e-3
noise_G = gaussian_noise_generator(sigma)
#func = lambda x, order=0: sphere_function(x, 2, order)
func = lambda x, order=0: bell_curve(x, order)
#func = weird_func
#func = boyd_example_func
H = np.matrix('10 0; 0 40')
# the vector of linear coefficient of the quadratic function
b = np.matrix('0; 0')
#func = lambda x, order=0: quadratic(H, b,x, order)
func = lambda x, order=0: nesterov_function(x, order)
noisy_func = lambda x, n, explicit_noise=None: noisy_function(func, x, noise_G, n, explicit_noise = explicit_noise, noise_mode="multiply")
initial_x = np.matrix('10;10')
N = 4000
t = 5e-3
t1 = 5e-3
m =100
n = initial_x.shape[0]
L = compute_L(func, n, 100)
#L = 1
print(L)
direction_G1 = sphere_point_generator(n)
direction_G2 = gaussian_point_generator(n)
x0, xs0,_,_ = alg.rdfds(noisy_func, initial_x, L, m, t, N, direction_generator = direction_G1)
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
x2, xs2,time2,_ = alg.rg(noisy_func, initial_x, L, 1, t, N,  direction_generator = direction_G2)
#print("final x\n", x1)
x3, xs3,time2, _ = alg.stars(noisy_func, initial_x, L, 1, t, noise_generator=noise_G, maximum_iterations=N,  direction_generator = direction_G2)
print("final x\n", x2)
print("final value\n", bell_curve(x2))
fig = plt.figure()
plot_real_values(func, xs0, fig, "1")
plot_real_values(func, xs2, fig, "2")
plot_real_values(func, xs3, fig, "3")
plt.legend()


fig1 = plt.figure()
x=np.arange(-5, 5.1, 0.05)
y=np.arange(-5, 5.1, 0.05)
levels=np.arange(5, 1000, 10)
Z = np.zeros((len(x), len(y)))
for i in range(len(x)):
    for j in range(len(y)):
        Z[i, j] = func( np.matrix([x[i],y[j]]).T , 0 )

plt.contour( x, y, Z.T, levels, colors='0.75')
#plt.ion()
plt.draw()
plt.show()

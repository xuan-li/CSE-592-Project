import numpy as np
from scipy.linalg import eigh as largest_eigh
from numpy import exp


def bell_curve(x, order=0):
    x = np.asmatrix(x)
    if order == 0:
        if x.T * x < 1e-100:
            return 0
        value = np.exp(-1 / (x.T * x))
        return value
    elif order == 1:
        if x.T * x < 1e-100:
            return (0, 0)
        value = np.exp(-1 / (x.T * x))
        gradient = (2 * value[0, 0] / np.power((x.T * x))[0, 0], 2) * x
        return (value, gradient)
    elif order == 2:
        if x.T * x < 1e-100:
            return (0, 0, 0)
        value = np.exp(-1 / (x.T * x))
        gradient = None
        S = np.matrix([[-x[1, 0] / x[0, 0], x[0, 0] / x[1, 0]], [1, 1]])
        J = np.matrix(
            [
                [2 * value[0, 0] / np.power((x.T * x)[0, 0], 2), 0], [
                    0, value[0, 0] * (
                        -6 / np.power((x.T * x)[0, 0], 2) +
                        4 / np.power((x.T * x)[0, 0], 3)
                    )
                ]
            ]
        )
        hessian = S * J * S.I
        return value, gradient, hessian


def boyd_example_func(x, order=0):

    a = np.matrix('1  3')
    b = np.matrix('1  -3')
    c = np.matrix('-1  0')
    x = np.asmatrix(x)

    value = exp(a * x - 0.1) + exp(b * x - 0.1) + exp(c * x - 0.1)
    if order == 0:
        return value
    elif order == 1:
        gradient = a.T * exp(a * x - 0.1) + b.T * exp(b * x - 0.1
                                                     ) + c.T * exp(c * x - 0.1)
        return (value, gradient)
    elif order == 2:
        gradient = a.T * exp(a * x - 0.1)[0, 0] + b.T * exp(b * x - 0.1)[
            0, 0
        ] + c.T * exp(c * x - 0.1)[0, 0]
        hessian = a.T * a * exp(a * x - 0.1)[0, 0] + b.T * b * exp(b * x - 0.1)[
            0, 0
        ] + c.T * c * exp(c * x - 0.1)[0, 0]
        return (value, gradient, hessian)
    else:
        raise ValueError("The argument \"order\" should be 0, 1 or 2")


def quadratic(H, b, x, order=0):
    """ 
    Quadratic Objective
    H:          the Hessian matrix
    b:          the vector of linear coefficients
    x:          the current iterate
    order:      the order of the oracle. For example, order=1 returns the value of the function and its gradient while order=2 will also return the hessian
    """
    H = np.asmatrix(H)
    b = np.asmatrix(b)
    x = np.asmatrix(x)

    value = 0.5 * x.T * H * x + b.T * x

    if order == 0:
        return value
    elif order == 1:
        gradient = H * x + b
        return (value, gradient)
    elif order == 2:
        gradient = H * x + b
        hessian = H
        return (value, gradient, hessian)
    else:
        raise ValueError("The argument \"order\" should be 0, 1 or 2")


def sphere_function(x, n, order=0):
    x = np.asmatrix(x)
    value = 0.5 * (x - 0.2).T * (x - 0.2)
    if order == 0:
        return value
    elif order == 1:
        gradient = x - 0.2
        return (value, gradient)
    elif order == 2:
        gradient = x - 0.2
        hessian = np.identity(n)
        return (value, gradient, hessian)
    else:
        raise ValueError("The argument \"order\" should be 0, 1 or 2")


def nesterov_function(x, order=0):
    x = np.asmatrix(x)
    y = np.vstack((x, np.zeros((1, 1))))
    z = np.vstack((np.zeros((1, 1)), x))
    diff = (z - y)
    value = diff.T * diff * 0.5 - x[0, 0]
    if order == 0:
        return np.asmatrix(value)
    elif order == 2:
        diag1 = np.ones(x.shape[0])
        diag2 = np.ones(x.shape[0] - 1)
        hessian = 2 * np.diag(diag1) - np.diag(diag2, 1) - np.diag(diag2, -1)

        return (value, None, hessian)


def noisy_values(
    func, x, noise_generator, n, explicit_noise=None, noise_mode="add"
):
    '''
    This functions assumes that we have one-point feedback, 
      that is, with one noise, we can only evaluate at one points

    func: exact function.
    noise_mode: add or multiply
        add : compute f(x) + noise
        multiply: compute f(x)(1 + noise)
    n: evaluating times.

    return shape (1,n)
    '''
    value = func(x)[0, 0]
    if not explicit_noise:
        noise = noise_generator(n)
    else:
        noise = explicit_noise
    if (noise_mode == "add"):
        value += noise
    if (noise_mode == "multiply"):
        value *= (1 + noise)
    noisy_values = np.asmatrix(value)

    return noisy_values


def noisy_paired_values(func, x, y, noise_generator, n, noise_mode="add"):
    '''
    This function assumes that we have two-point feedback, 
      that is, we use the same noise to evaluate at two points.

    func: exact function.
    noise_mode: add or multiply
        add : compute f(x) + noise
        multiply: compute f(x)(1 + noise)
    n: evaluating times.

    return shape (2,n)
    '''
    valuex = func(x)[0, 0]
    valuey = func(y)[0, 0]
    noise = noise_generator(n)

    if (noise_mode == "add"):
        valuex += noise
        valuey += noise
    if (noise_mode == "multiply"):
        valuex *= (1 + noise)
        valuey *= (1 + noise)
    valuex = np.asmatrix(valuex)
    valuey = np.asmatrix(valuey)
    noisy_values = (valuex, valuey)
    return noisy_values


def noisy_function(
    func, x, noise_generator, n, explicit_noise=None, noise_mode="add"
):
    if "list" not in str(type(x)):
        return noisy_values(
            func,
            x,
            noise_generator,
            n,
            explicit_noise=explicit_noise,
            noise_mode=noise_mode
        )
    else:
        assert (len(x) == 2)
        return noisy_paired_values(
            func, x[0], x[1], noise_generator, n, noise_mode
        )


def compute_L(func, n, bound=100):
    # https://math.stackexchange.com/questions/1698812/lipschitz-constant-gradient-implies-bounded-eigenvalues-on-hessian
    L = 0

    def update_L(func, x, L, beta=0.9):
        value, gradient, hessian = func(x, 2)
        # largest eigen value
        # https://stackoverflow.com/questions/12167654/fastest-way-to-compute-k-largest-eigenvalues-and-corresponding-eigenvectors-with
        evals_large, _ = largest_eigh(hessian, eigvals=(n - 1, n - 1))
        evals_small, _ = largest_eigh(hessian, eigvals=(0, 0))
        return max([L, abs(evals_large[0]), abs(evals_small[0])])

    size = 10000
    rvs = np.asmatrix(np.random.uniform(low=-bound, high=bound, size=(n, size)))
    for i in range(size):
        L = update_L(func, rvs[:, i], L)
    return L


if __name__ == '__main__':
    x = np.matrix("[1;2;3;4;5;6]")
    print(nesterov_function(x, 2))

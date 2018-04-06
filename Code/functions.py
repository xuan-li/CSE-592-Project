import numpy as np



def weird_func( x, order=0 ):

    # f(x) = x^4 + 6x^2 + 12(x-4)e^(x-1)
    value = pow(x, 4) + 6 * pow(x, 2) + 12 * (x - 4) * exp(x - 1)
    
    if order==0:
        return value
    elif order==1:
        # f'(x) = 4x^3 + 12x + 12(x-3)e^(x-1)
        gradient = 4 * pow(x, 3) + 12 * x + 12 * (x - 3) * exp(x - 1)

        return (value, gradient)
    elif order==2:
        # f'(x) = 4x^3 + 12x + 12(x-3)e^(x-1)
        gradient = 4 * pow(x, 3) + 12 * x + 12 * (x - 3) * exp(x - 1)

        # f''(x)= 12 (1 + e^(-1 + x) (-2 + x) + x^2)
        hessian = 12 * (1 + (x-2) * exp(x-1) + pow(x,2))

        return (value, gradient, hessian)
    else:
        raise ValueError("The argument \"order\" should be 0, 1 or 2")


def boyd_example_func(x, order=0):
    a=np.matrix('1  3')
    b=np.matrix('1  -3')
    c=np.matrix('-1  0')
    x=np.asmatrix(x)
    
    value = exp(a*x-0.1)+exp(b*x-0.1)+exp(c*x-0.1)
    if order==0:
        return value
    elif order==1:
        gradient = a.T*exp(a*x-0.1)+b.T*exp(b*x-0.1)+c.T*exp(c*x-0.1)
        return (value, gradient)
    elif order==2:
        gradient = a.T*exp(a*x-0.1)+b.T*exp(b*x-0.1)+c.T*exp(c*x-0.1)
        hessian = a.T*a*exp(a*x-0.1)+b.T*b*exp(b*x-0.1)+c.T*c*exp(c*x-0.1)
        return (value, gradient, hessian)
    else:
        raise ValueError("The argument \"order\" should be 0, 1 or 2")


def quadratic( H, b, x, order=0 ):
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

def sphere_function(x, n, order = 0):
    x = np.asmatrix(x)
    value = 0.5 * (x - 0.2).T * (x-0.2)
    if order == 0:
        return value
    elif order == 1:
        gradient = x-0.2
        return (value, gradient)
    elif order == 2:
        gradient = x-0.2
        hessian = H
        return (value, gradient, hessian)
    else:
        raise ValueError("The argument \"order\" should be 0, 1 or 2")


def noisy_values(func, x, noise_generator, n, noise_mode="add"):
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
    value = func(x)[0,0]
    noise = noise_generator(n)
    if (noise_mode == "add"):
        value += noise
    if (noise_mode == "multiply"):
        value *= (1+noise)
    noisy_values = np.asmatrix(value)

    return noisy_values


def noisy_paired_values(func, x,y, noise_generator, n, noise_mode="add"):
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
    valuex = func(x)[0,0]
    valuey = func(y)[0,0]
    noise = noise_generator(n)
    
    if (noise_mode == "add"):
        valuex += noise
        valuey += noise
    if (noise_mode == "multiply"):
        valuex *= (1+noise)
        valuey *= (1+noise)
    valuex = np.asmatrix(valuex)
    valuey = np.asmatrix(valuey)
    noisy_values = (valuex, valuey)
    return noisy_values


def noisy_function(func, x, noise_generator, n, noise_mode="add"):
    if "list" not in str(type(x)):
        return noisy_values(func, x, noise_generator,n,noise_mode)
    else :
        assert(len(x) == 2)
        return noisy_paired_values(func, x[0],x[1], noise_generator,n,noise_mode )










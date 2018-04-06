import numpy as np

def sphere_point_generator(ndim):
    '''
    generate npoints vectors on the Euclidean unit sphere in R^ndim
    '''
    def sample_spherical(npoints):
        vec = np.random.randn(ndim, npoints)
        vec /= np.linalg.norm(vec, axis=0)
        return np.asmatrix(vec)
    return lambda npoints: sample_spherical(npoints)

def gaussian_noise_generator(scale):
    def sample_spherical(npoints):
        vec = np.random.normal(loc=0, scale=scale, size=npoints)
        return np.asmatrix(vec)
    return lambda npoints: sample_spherical(npoints)

def test():
    generator = sphere_point_generator(10)
    print(generator(2))
    generator = gaussian_noise_generator(0.1)
    print(generator(3))

if __name__ == '__main__':
    test()
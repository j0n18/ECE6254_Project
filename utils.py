import numpy as np

def sample_annulus(n_samples, r_outer, r_inner, seed = None):
    '''
    Sample values that lie between two radii.
    '''

    if seed is not None:
        np.random.seed(seed)

    rho = np.sqrt(np.random.uniform(r_inner**2, r_outer**2, n_samples))
    theta = np.random.uniform(0, 2*np.pi, n_samples)
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)

    return x, y
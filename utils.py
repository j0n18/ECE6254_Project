import numpy as np
from ECE6254_Project.models.VAE import VAE
from ECE6254_Project.models.AE import AE
from ECE6254_Project.models.CVAE import CVAE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib import colors
import torch


def sample_annulus(n_samples, r_outer, r_inner, seed=None):
    '''
    Sample values that lie between two radii.
    '''

    if seed is not None:
        np.random.seed(seed)

    rho = np.sqrt(np.random.uniform(r_inner ** 2, r_outer ** 2, n_samples))
    theta = np.random.uniform(0, 2 * np.pi, n_samples)
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)

    return x, y


def load_pretrained_model(model, chkpt):
    if model == "VAE":
        trained_model = VAE.load_from_checkpoint(chkpt)
    elif model == "AE":
        trained_model = AE.load_from_checkpoint(chkpt)
    elif model == "CVAE":
        trained_model = CVAE.load_from_checkpoint(chkpt)

    trained_model.freeze()

    return trained_model


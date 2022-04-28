import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib
import torch


def plot_valid_and_recon(valid_data, valid_labels, trained_model, model_label, plot_title="valid"):
    colors = ['r', 'b']

    x_valid = valid_data[:, 0]
    y_valid = valid_data[:, 1]

    rec, z = trained_model.forward(valid_data)

    x_rec_valid = rec[:, 0]
    y_rec_valid = rec[:, 1]

    # plot valid_data:
    plt.scatter(x_valid,
                y_valid,
                c=valid_labels,
                cmap=matplotlib.colors.ListedColormap(colors))

    plt.title(plot_title)
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.show()

    # plot reconstruction of valid_data:
    plt.scatter(x_rec_valid,
                y_rec_valid,
                c=valid_labels,
                cmap=matplotlib.colors.ListedColormap(colors))

    plt.title(f'reconstruction ({model_label})')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.show()


def plot_latents(valid_data, valid_labels, trained_model, model_label, colormap=None, show=True, legend=False,
                 title=None):
    if colormap is None:
        colormap = matplotlib.colors.ListedColormap(['r', 'b'])

    if title is None:
        title = f"$\phi(z)$ (from {model_label})"

    _, z = trained_model.forward(valid_data)

    pca = PCA(n_components=3)
    z_pca = pca.fit_transform(z)

    if show:
        fig = plt.figure()
        ax = plt.axes(projection='3d')

        scatter = ax.scatter(
            z_pca[:, 0],
            z_pca[:, 1],
            z_pca[:, 2],
            c=valid_labels,
            cmap=colormap
        )
        ax.set_title(title)

        if legend:
            legend1 = ax.legend(*scatter.legend_elements(),
                                loc="upper right", title="Classes")
            ax.add_artist(legend1)

        plt.show()

    return z_pca


def plot_2D_decision_boundary(classifier, data, labels, classifier_label, model_label, mesh_stepsize=0.02):
    '''
    Function for plotting decision boundaries. Pass in the top 2 PCs as input to get 2D decision.

    Credit:
    https://mdav.ece.gatech.edu/ece-6254-spring2022/assignments/knn-example.py
    '''

    ## Plot the decision boundary. 
    # Begin by creating the mesh [x_min, x_max]x[y_min, y_max].
    x_delta = (data[:, 0].max() - data[:, 0].min()) * 0.05  # add 5% white space to border
    y_delta = (data[:, 1].max() - data[:, 1].min()) * 0.05
    x_min, x_max = data[:, 0].min() - x_delta, data[:, 0].max() + x_delta
    y_min, y_max = data[:, 1].min() - y_delta, data[:, 1].max() + y_delta
    xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_stepsize), np.arange(y_min, y_max, mesh_stepsize))

    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])

    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#0000FF'])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    ## Plot the training points
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap=cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(f'{classifier_label} decision boundary; features from {model_label}')

    ## Show the plot
    plt.show()


def plot_3D_decision_boundary(svc, data, labels, classifier_label, model_label, mesh_stepsize=0.02):
    '''
    Function for plotting decision boundaries. 

    Heavy influence on the development of this function:
    https://stackoverflow.com/questions/36232334/plotting-3d-decision-boundary-from-linear-svm
    https://mdav.ece.gatech.edu/ece-6254-spring2022/assignments/knn-example.py
    https://stackoverflow.com/questions/21418255/changing-the-line-color-in-plot-surface

    '''

    z = lambda x, y: (-svc.intercept_[0] - svc.coef_[0][0] * x - svc.coef_[0][1] * y) / svc.coef_[0][2]

    ## Plot the decision boundary. 
    # Begin by creating the mesh [x_min, x_max]x[y_min, y_max].
    x_delta = (data[:, 0].max() - data[:, 0].min()) * 0.05  # add 5% white space to border
    y_delta = (data[:, 1].max() - data[:, 1].min()) * 0.05
    x_min, x_max = data[:, 0].min() - x_delta, data[:, 0].max() + x_delta
    y_min, y_max = data[:, 1].min() - y_delta, data[:, 1].max() + y_delta
    xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_stepsize), np.arange(y_min, y_max, mesh_stepsize))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xx, yy, z(xx, yy), alpha=0.5, linewidth=0.5, edgecolors='grey')
    ax.plot3D(data[labels == 0, 0], data[labels == 0, 1], data[labels == 0, 2], 'or', alpha=0.5)
    ax.plot3D(data[labels == 1, 0], data[labels == 1, 1], data[labels == 1, 2], 'ob', alpha=0.5)
    plt.title(f'{classifier_label} decision boundary; features from {model_label}')
    plt.show()


""" Plotting functions for image data 

# visualizing the original images vs their latents 
1. first 3 PCs of the original image data 
3. labelled scatter plot of the latents learned by the vae 
2. first 3 PCs of the latents learned by the vae on the original image data 

# visualizing the performance of a linear SVM for the original image data vs their latents 
"""


def plot_imgs_on_grid(imgs, inds=None, fig_title=None, im_labels=None, label_dict=None, grid_dim=(3, 3)):
    """
    plots a sample of the imgs on a grid defined by grid_dim
    :param imgs: should be a tensor with shape (num_imgs, num_channels, im_width, im_height)
    :param inds: (ndarray) indices of the images to be plotted - must be the same lengeth as grid_dim[0]*grid_dim[1]
    :param fig_title: title of the figure
    :param im_labels: labels to be used as titles on the images
    :param label_dict: dictionary that maps im_labels to label strings with im_labels being integers
    :param grid_dim: dimension of the grid on which the images are to be plotted
    :return:
    """
    cols, rows = grid_dim
    num_channels = imgs.shape[1]
    if inds is None:
        inds = np.random.choice(np.arange(0, imgs.shape[0]), size=cols * rows)
    sample_imgs = imgs[inds, :, :, :].squeeze().detach()

    if im_labels is not None:
        if label_dict is None:
            im_titles = [str(lbl.item()) for lbl in im_labels[inds]]
        else:
            im_titles = [label_dict[lbl] for lbl in im_labels[inds]]
    else:
        im_titles = [""] * len(inds)

    fig = plt.figure(figsize=(8, 8))

    if fig_title is not None:
        plt.suptitle(fig_title)

    for i in range(1, cols * rows + 1):
        fig.add_subplot(rows, cols, i)
        plt.title(im_titles[i - 1])
        plt.axis("off")
        if num_channels > 1:
            plt.imshow(sample_imgs[i - 1, :, :, :].permute(1, 2, 0))
        else:
            plt.imshow(sample_imgs[i - 1, :, :], cmap='gray')


def plot_valid_recon_mnist(valid_data, valid_labels, trained_model, model_name="CVAE"):
    """
    plots two figures showing sample original images and their corresponding reconstructions
    :param valid_data:
    :param valid_labels:
    :param trained_model:
    :param model_name:
    :return:
    """
    recon_data = trained_model(valid_data)[0]
    cols, rows = 3, 3
    # indices for sample images to be plotted
    inds = np.random.choice(np.arange(0, valid_data.shape[0]), size=cols * rows)
    # plot the original data
    plot_imgs_on_grid(valid_data, inds=inds, fig_title="Original MNIST Examples",
                      im_labels=valid_labels, grid_dim=(rows, cols))
    # plot the reconstructed data
    plot_imgs_on_grid(recon_data, inds=inds, fig_title=f"Reconstruced MNIST Examples ({model_name})",
                      im_labels=valid_labels, grid_dim=(rows, cols))

    plt.show()


def plot_generated_mnist(trained_model):
    """
    plots some generated examples by the trained model
    :param trained_model:
    :return:
    """

    latent_dim = trained_model.hparams['latent_dim']
    z = torch.randn(9, latent_dim)
    generated_data = trained_model.decoder(z)
    plot_imgs_on_grid(generated_data, fig_title="MNIST Examples Generated By the CVAE Model",
                      grid_dim=(3, 3))
    plt.show()


def scatter_mnist_PCs(imgs, labels, colormap, show=True, legend=False,
                      title=None):
    if title is None:
        title = f"Top Three Principal Components of the Raw Image Data"

    imgs = imgs.squeeze().detach()
    imgs = np.reshape(imgs, (imgs.shape[0], imgs.shape[1] * imgs.shape[2]))

    pca = PCA(n_components=3)
    im_pca = pca.fit_transform(imgs)

    if show:
        fig = plt.figure()
        ax = plt.axes(projection='3d')

        scatter = ax.scatter(
            im_pca[:, 0],
            im_pca[:, 1],
            im_pca[:, 2],
            c=labels,
            cmap=colormap
        )
        ax.set_title(title)

        if legend:
            legend1 = ax.legend(*scatter.legend_elements(),
                                loc="upper right", title="Classes")
            ax.add_artist(legend1)

        plt.show()

    return im_pca

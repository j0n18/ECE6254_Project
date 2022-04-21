import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib
from sklearn.metrics import ConfusionMatrixDisplay

def plot_valid_and_recon(valid_data, valid_labels, trained_model, model_label, plot_title = "valid"):

    colors = ['r', 'b']

    x_valid = valid_data[:,0]
    y_valid = valid_data[:,1]

    rec, z = trained_model.forward(valid_data)

    x_rec_valid = rec[:,0]
    y_rec_valid = rec[:,1]

    #plot valid_data:
    plt.scatter(x_valid, 
                y_valid, 
                c = valid_labels, 
                cmap=matplotlib.colors.ListedColormap(colors))

    plt.title(plot_title)
    plt.xlim(-5,5)
    plt.ylim(-5,5)
    plt.show()

    #plot reconstruction of valid_data:
    plt.scatter(x_rec_valid, 
                y_rec_valid, 
                c = valid_labels, 
                cmap=matplotlib.colors.ListedColormap(colors))

    plt.title(f'reconstruction ({model_label})')
    plt.xlim(-5,5)
    plt.ylim(-5,5)
    plt.show()


def plot_latents(valid_data, valid_labels, trained_model, model_label, show = True):

    colors = ['r', 'b']

    _, z = trained_model.forward(valid_data)

    pca = PCA(n_components=3)
    z_pca = pca.fit_transform(z)

    if show:
        fig = plt.figure()
        ax = plt.axes(projection ='3d')

        ax.scatter(
            z_pca[:, 0],
            z_pca[:, 1],
            z_pca[:, 2],
            c = valid_labels,
            cmap=matplotlib.colors.ListedColormap(colors)
        )
        ax.set_title(f"$\phi(z)$ (from {model_label})")


        plt.show()

    return z_pca


def plot_2D_decision_boundary(classifier, data, labels, classifier_label, model_label, mesh_stepsize = 0.02):
    '''
    Function for plotting decision boundaries. Pass in the top 2 PCs as input to get 2D decision.

    Credit:
    https://mdav.ece.gatech.edu/ece-6254-spring2022/assignments/knn-example.py
    '''

    ## Plot the decision boundary. 
    # Begin by creating the mesh [x_min, x_max]x[y_min, y_max].
    x_delta = (data[:, 0].max() - data[:, 0].min())*0.05 # add 5% white space to border
    y_delta = (data[:, 1].max() - data[:, 1].min())*0.05
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


def plot_3D_decision_boundary(svc, data, labels,classifier_label, model_label, mesh_stepsize = 0.02):

    '''
    Function for plotting decision boundaries. 

    Heavy influence on the development of this function:
    https://stackoverflow.com/questions/36232334/plotting-3d-decision-boundary-from-linear-svm
    https://mdav.ece.gatech.edu/ece-6254-spring2022/assignments/knn-example.py
    https://stackoverflow.com/questions/21418255/changing-the-line-color-in-plot-surface

    '''

    z = lambda x,y: (-svc.intercept_[0]-svc.coef_[0][0]*x-svc.coef_[0][1]*y) / svc.coef_[0][2]

    ## Plot the decision boundary. 
    # Begin by creating the mesh [x_min, x_max]x[y_min, y_max].
    x_delta = (data[:, 0].max() - data[:, 0].min())*0.05 # add 5% white space to border
    y_delta = (data[:, 1].max() - data[:, 1].min())*0.05
    x_min, x_max = data[:, 0].min() - x_delta, data[:, 0].max() + x_delta
    y_min, y_max = data[:, 1].min() - y_delta, data[:, 1].max() + y_delta
    xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_stepsize), np.arange(y_min, y_max, mesh_stepsize))

    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xx, yy, z(xx,yy), alpha = 0.5, linewidth = 0.5, edgecolors='grey')
    ax.plot3D(data[labels==0,0], data[labels==0,1], data[labels==0,2],'or', alpha = 0.5)
    ax.plot3D(data[labels==1,0], data[labels==1,1], data[labels==1,2],'ob', alpha = 0.5)
    plt.title(f'{classifier_label} decision boundary; features from {model_label}')
    plt.show()

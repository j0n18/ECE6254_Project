import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def plot_valid_and_recon(valid_data, trained_model, model_label):

    x_valid = valid_data[:,0]
    y_valid = valid_data[:,1]

    rec, z = trained_model.forward(valid_data)

    x_rec_valid = rec[:,0]
    y_rec_valid = rec[:,1]

    #plot valid_data:
    plt.scatter(x_valid, y_valid, label = 'valid')
    plt.xlim(-5,5)
    plt.ylim(-5,5)
    plt.legend()
    plt.show()

    #plot reconstruction of valid_data:
    plt.scatter(x_rec_valid, y_rec_valid, label = f'reconstruction ({model_label})')
    plt.xlim(-5,5)
    plt.ylim(-5,5)
    plt.legend()
    plt.show()


def plot_latents(valid_data, trained_model, model_label):

    _, z = trained_model.forward(valid_data)

    pca = PCA(n_components=3)
    z_pca = pca.fit_transform(z)

    fig = plt.figure()
    ax = plt.axes(projection ='3d')

    ax.scatter(
        z_pca[:, 0],
        z_pca[:, 1],
        z_pca[:, 2],
        "b",
        label=f"$$\phi(z)$$",
    )
    ax.set_title(f"$\phi(z)$ (from {model_label}")

    plt.show()

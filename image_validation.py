import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from torch.utils.data import DataLoader
from plotting import (plot_latents,
                      plot_valid_recon_mnist,
                      plot_generated_mnist,
                      scatter_mnist_PCs)

from mnist_datamodule import MNISTDataModule
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from utils import load_pretrained_model


def img_tensor_to_numpy_vector(t):
    return np.reshape(t.squeeze().detach(), (t.shape[0], np.prod(t.shape[1:])))


log = logging.getLogger(__name__)

chkpt_cvae = ".\\cvae_chkpts\\last.ckpt"

mnist = MNISTDataModule()

mnist.prepare_data()
mnist.setup()

train_data, train_labels = next(iter(DataLoader(mnist.mnist_train, batch_size=len(mnist.mnist_train))))
valid_data, valid_labels = next(iter(DataLoader(mnist.mnist_val, batch_size=len(mnist.mnist_val))))
test_data, test_labels = next(iter(DataLoader(mnist.mnist_test, batch_size=len(mnist.mnist_test))))

# load the models
cvae_model = load_pretrained_model("CVAE", chkpt_cvae)

# visualize the original and reconstructed images
plot_valid_recon_mnist(valid_data, valid_labels, cvae_model)

# visualize the model's generative examples
plot_generated_mnist(cvae_model)

# plot the latents
z_pca_cvae = plot_latents(train_data, train_labels, cvae_model, 'CVAE',
                          colormap=cm.get_cmap('tab10', 10),
                          legend=True,
                          title=f"Latent Representations of MNIST Data Learned by the CVAE")

# plot the first 3 pcs of the original data for comparison
train_pcs = scatter_mnist_PCs(train_data, train_labels, cm.get_cmap('tab20', 10),
                              legend=True)
test_pcs = scatter_mnist_PCs(test_data, test_labels, cm.get_cmap('tab20', 10), show=False)

# fit linear classifiers to the raw image data
# first transform the data into flattened numpy arrays
"""
train_data_numpy = img_tensor_to_numpy_vector(train_data)
test_data_numpy = img_tensor_to_numpy_vector(test_data)
train_labels_numpy = train_labels.detach()
test_labels_numpy = test_labels.detach()
"""

# get the image latents
train_latents = cvae_model(train_data)[1]
test_latents = cvae_model(test_data)[1]

# construct the classifiers
svc = SVC(kernel='linear')
lr = LogisticRegression()
lda = LinearDiscriminantAnalysis()

# fit
svc_pcs = svc.fit(train_pcs, train_labels.detach())
svc_latents = svc.fit(train_latents, train_labels.detach())
lr_pcs = lr.fit(train_pcs, train_labels.detach())
lr_latents = lr.fit(train_latents, train_labels.detach())
lda_pcs = lda.fit(train_pcs, train_labels.detach())
lda_latents = lda.fit(train_latents, train_labels.detach())

# confusion matrix
models = ["SVC", "Logistic Regression", "Linear Discrminiant Analysis"]
cnt = 0
for mdl_pcs, mdl_latents in zip([svc_pcs, lr_pcs, lda_pcs], [svc_latents, lr_latents, lda_latents]):
    plot_confusion_matrix(mdl_pcs, test_pcs, test_labels.detach(), normalize='true', cmap=cm.Blues)
    plt.title(f"{models[cnt]} Confusion Matrix for the Principal Components of the Raw Images")
    plot_confusion_matrix(mdl_latents, test_latents, test_labels.detach(), normalize='true', cmap=cm.Blues)
    plt.title(f"{models[cnt]} Confusion Matrix for the CVAE Latents")
    cnt += 1

plt.show()

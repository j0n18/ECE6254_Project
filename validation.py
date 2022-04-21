import logging
from params import dm_params
import matplotlib.pyplot as plt

from plotting import (plot_3D_decision_boundary, 
                        plot_2D_decision_boundary, 
                        plot_latents, 
                        plot_valid_and_recon)

from synthetic_datamodule import SyntheticDataModule
from sklearn.svm import SVC
from sklearn.metrics import plot_confusion_matrix

from utils import load_pretrained_model

log = logging.getLogger(__name__)

#import pdb; pdb.set_trace();

chkpt_vae = "C:\\Users\\jonat\\OneDrive\\Desktop\\StatML\\ECE6254_Project\\vae_chkpts\\good_runs\\best\\epoch=50-step=1326.ckpt"
#"C:\\Users\\jonat\\OneDrive\\Desktop\\StatML\\ECE6254_Project\\vae_chkpts\\epoch=50-step=1326.ckpt"

chkpt_ae = "C:\\Users\\jonat\\OneDrive\\Desktop\\StatML\\ECE6254_Project\\ae_chkpts\\good_runs\\best\\epoch=249-step=6500.ckpt"
#"C:\\Users\\jonat\\OneDrive\\Desktop\\StatML\\ECE6254_Project\\ae_chkpts\\epoch=99-step=2600.ckpt"


#Instantiate the datamodule:
synthetic_datamodule = SyntheticDataModule(**dm_params)

# run prepare_data to load/generate the data set:
synthetic_datamodule.prepare_data()
synthetic_datamodule.setup()

#get training set:
train_data, train_labels, _ = synthetic_datamodule.train_ds.tensors

#get validation set:
valid_data, valid_labels, _ = synthetic_datamodule.valid_ds.tensors

#get test set for evaluation metrics:
test_data, test_labels, _ = synthetic_datamodule.test_ds.tensors

#x_test = train_data[:,0]
#y_test = train_data[:,1]

#load the pretrained models:
vae = load_pretrained_model("VAE", chkpt_vae)
ae = load_pretrained_model("AE", chkpt_ae)

#plot the original validation data and the reconstructions:
plot_valid_and_recon(train_data, train_labels, vae, "VAE", plot_title = "train")
plot_valid_and_recon(train_data, train_labels, ae, "AE", plot_title = "train")

#plot the top 3 PCs of the latents:
z_pca_vae = plot_latents(train_data, train_labels, vae, "VAE")
z_pca_ae = plot_latents(train_data, train_labels, ae, "AE")

#get test data latents for later evaluation:
z_pca_vae_test = plot_latents(test_data, test_labels, vae, "VAE", show = False)
z_pca_ae_test = plot_latents(test_data, test_labels, ae, "AE", show = False)

#get the recons and latents for both the VAE and AE:
#recon_vae, z_vae = vae.forward(test_data)
#recon_ae, z_ae = ae.forward(test_data)

#Create a linear classifier and train it on the original and transformed data:
svc = SVC(kernel='linear')

#fit classifier and plot decision boundary in corresponding space:
svc_regular =  svc.fit(train_data, train_labels)
plot_2D_decision_boundary(svc_regular, train_data, train_labels, "SVC", "Regular Data")

#Calculate and plot classification metrics:
plot_confusion_matrix(svc_regular, test_data, test_labels, normalize = "true", cmap=plt.cm.Blues)
plt.title("Regular Data")
plt.show()

svc_vae =  svc.fit(z_pca_vae, train_labels)
plot_3D_decision_boundary(svc_vae, z_pca_vae, train_labels, "SVC", "VAE")

#Calculate and plot classification metrics:
plot_confusion_matrix(svc_vae, z_pca_vae_test, test_labels, normalize = "true", cmap=plt.cm.Blues)
plt.title("VAE")
plt.show()

svc_ae =  svc.fit(z_pca_ae, train_labels)
plot_3D_decision_boundary(svc_ae, z_pca_ae, train_labels, "SVC", "AE")

#Calculate and plot classification metrics:
plot_confusion_matrix(svc_ae, z_pca_ae_test, test_labels, normalize = "true", cmap=plt.cm.Blues)
plt.title("AE")
plt.show()

import pdb; pdb.set_trace();

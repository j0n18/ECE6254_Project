import logging
from params import dm_params

from plotting import (plot_3D_decision_boundary, 
                        plot_2D_decision_boundary,
                        plot_latents, 
                        plot_valid_and_recon)

from synthetic_datamodule import SyntheticDataModule
from sklearn.svm import SVC
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

#get validation set:
valid_data, valid_labels, _ = synthetic_datamodule.valid_ds.tensors

x_valid = valid_data[:,0]
y_valid = valid_data[:,1]

#load the pretrained models:
vae = load_pretrained_model("VAE", chkpt_vae)
ae = load_pretrained_model("AE", chkpt_ae)

#plot the original validation data and the reconstructions:
plot_valid_and_recon(valid_data, valid_labels, vae, "VAE")
plot_valid_and_recon(valid_data, valid_labels, ae, "AE")


#plot the top 3 PCs of the latents:
z_pca_vae = plot_latents(valid_data,valid_labels, vae, "VAE")
z_pca_ae = plot_latents(valid_data, valid_labels, ae, "AE")

#get the recons and latents for both the VAE and AE:
recon_vae, z_vae = vae.forward(valid_data)
recon_ae, z_ae = ae.forward(valid_data)

#Create a linear classifier and train it on the original and transformed data:
svc = SVC(kernel='linear')

#fit classifier and plot decision boundary in corresponding space:
svc_regular =  svc.fit(valid_data, valid_labels)
plot_2D_decision_boundary(svc_regular, valid_data, valid_labels, "SVC", "Regular Data")

svc_vae =  svc.fit(z_pca_vae, valid_labels)
plot_3D_decision_boundary(svc_vae, z_pca_vae, valid_labels, "SVC", "VAE")

svc_ae =  svc.fit(z_pca_ae, valid_labels)
plot_3D_decision_boundary(svc_ae, z_pca_ae, valid_labels, "SVC", "AE")

import pdb; pdb.set_trace();


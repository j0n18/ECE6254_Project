import logging

import numpy as np
import torch
from pytorch_lightning import Trainer
from torchvision import transforms

from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from ECE6254_Project.models.VAE import VAE
from ECE6254_Project.models.AE import AE
from ECE6254_Project.models.CVAE import CVAE
from ECE6254_Project.models.CAE import CAE
from synthetic_datamodule import SyntheticDataModule
from mnist_datamodule import MNISTDataModule

from params import (dm_params,
                    model_params_vae,
                    model_params_ae,
                    model_ckpt_params_vae,
                    model_ckpt_params_cvae,
                    model_ckpt_params_ae,
                    model_ckpt_params_cae,
                    early_stopping_params,
                    lr_monitor_params,
                    csv_logger_params,
                    tensorboard_logger_params,
                    trainer_params)


log = logging.getLogger(__name__)

# Need to instantiate the following:
# 1. datamodule (lightning data module)
# 2. model (lightning module)
# 3. callbacks
# 4. loggers
# 5. trainer

# instantiate datamodules:
datamodule = SyntheticDataModule(**dm_params)
mnist = MNISTDataModule()
mnist_mlp = MNISTDataModule()
mnist_mlp.transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(torch.flatten)
])

# instantiate model:
log.info("Instantiating VAEs...")
vae_model = VAE(**model_params_vae)
vae_model_im = VAE(input_dim=np.prod(mnist.dims),
                   hidden_size=50,
                   output_dim=3,
                   drop_prop=0.05)

log.info("Instantiating AE...")
ae_model = AE(**model_params_ae)

log.info("Instantiating CVAE...")
hidden_layers = [32, 64, 128, 256]
latent_dim = 3
cvae_model = CVAE(input_shape=mnist.dims,
                  hidden_layers=hidden_layers,
                  latent_dim=latent_dim)

log.info("Instantiating CAE...")
cae_model = CAE(input_shape=mnist.dims,
                hidden_layers=hidden_layers,
                latent_dim=latent_dim)

# instantiate the callbacks: (already done by the import)
log.info("Instantiating callbacks...")

callbacks_vae = [
    ModelCheckpoint(**model_ckpt_params_vae),
    EarlyStopping(**early_stopping_params),
    LearningRateMonitor(**lr_monitor_params)
]

callbacks_cvae = [
    ModelCheckpoint(**model_ckpt_params_cvae),
    EarlyStopping(**early_stopping_params),
    LearningRateMonitor(**lr_monitor_params)
]

callbacks_ae = [
    ModelCheckpoint(**model_ckpt_params_ae),
    EarlyStopping(**early_stopping_params),
    LearningRateMonitor(**lr_monitor_params)
]

callbacks_cae = [
    ModelCheckpoint(**model_ckpt_params_cae),
    EarlyStopping(**early_stopping_params),
    LearningRateMonitor(**lr_monitor_params)
]

# instantiate the loggers: (already done by the import)
log.info("Instantiating loggers...")

loggers = [
    CSVLogger(**csv_logger_params),
    TensorBoardLogger(**tensorboard_logger_params),
]

# instantiate trainer:
log.info("Instantiating Trainer...")

trainer_vae = Trainer(
    **trainer_params,
    callbacks=callbacks_vae,
    logger=loggers,
    accelerator='gpu',
    devices=1
)

trainer_cvae = Trainer(
    **trainer_params,
    callbacks=callbacks_cvae,
    logger=loggers,
    accelerator='gpu',
    devices=1
)

trainer_ae = Trainer(
    **trainer_params,
    callbacks=callbacks_ae,
    logger=loggers,
    accelerator='gpu',
    devices=1
)

trainer_cae = Trainer(
    **trainer_params,
    callbacks=callbacks_cae,
    logger=loggers,
    accelerator='gpu',
    devices=1
)

# Fit the trainer using the model and datamodules:
# log.info("Starting training VAE model.")
# trainer_vae.fit(model=vae_model, datamodule= datamodule)

# log.info("Starting training AE model.")
# trainer_ae.fit(model=ae_model, datamodule= datamodule)

log.info("Starting training the VAE model on MNIST.")
trainer_vae.fit(model=vae_model_im, datamodule=mnist_mlp)
print('done')

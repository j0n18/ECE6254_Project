import logging
from pytorch_lightning import Trainer

from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from ECE6254_Project.models.VAE import VAE
from ECE6254_Project.models.AE import AE
from synthetic_datamodule import SyntheticDataModule

from params import (dm_params, 
                    model_params_vae, 
                    model_params_ae,
                    model_ckpt_params_vae,
                    model_ckpt_params_ae,
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

# instantiate model:
log.info("Instantiating VAE...")
vae_model = VAE(**model_params_vae)

log.info("Instantiating VAE...")
ae_model = AE(**model_params_ae)

# instantiate the callbacks: (already done by the import)
log.info("Instantiating callbacks...")

callbacks_vae = [
    ModelCheckpoint(**model_ckpt_params_vae),
    EarlyStopping(**early_stopping_params),
    LearningRateMonitor(**lr_monitor_params)
]

callbacks_ae = [
    ModelCheckpoint(**model_ckpt_params_ae),
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
    logger=loggers
)

trainer_ae = Trainer(
    **trainer_params,
    callbacks=callbacks_ae,
    logger=loggers
)

# Fit the trainer using the model and datamodules:
log.info("Starting training VAE model.")
trainer_vae.fit(model=vae_model, datamodule= datamodule)

log.info("Starting training AE model.")
trainer_ae.fit(model=ae_model, datamodule= datamodule)
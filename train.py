import logging
from pytorch_lightning import Trainer

from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from ECE6254_Project.models.VAE import VAE
from synthetic_datamodule import SyntheticDataModule

log = logging.getLogger(__name__)

# Need to instantiate the following:
# 1. datamodule (lightning data module)
# 2. model (lightning module)
# 3. callbacks
# 4. loggers
# 5. trainer

# instantiate datamodules:

dm_params = {
    "dim": 2,
    "n_samples": 1000, 
    "r_outer1": 2, 
    "r_inner1": 0, 
    "r_outer2": 4, 
    "r_inner2": 3, 
    "seed": 0,
    "batch_size": 50
}

datamodule = SyntheticDataModule(**dm_params)

# instantiate model:

model_params = {
    "input_dim": 2, 
    "hidden_size": 4,
    "output_dim": 6, 
    "drop_prop": 0.05
}

log.info("Instantiating VAE...")
model = VAE(**model_params)

# instantiate the callbacks: (already done by the import)

log.info("Instantiating callbacks...")

model_ckpt_params = {
    "monitor": "total_loss",
    "mode": "min",
    "save_top_k": 1,
    "save_last": True,
    "verbose": False,
    "dirpath": ".",
    "auto_insert_metric_name": True,
}

early_stopping_params = {
    "monitor": "total_loss",
    "mode": "min",
    "patience": 200,
    "min_delta": 0,
}

lr_monitor_params = {"logging_interval": "epoch"}

callbacks = [
    ModelCheckpoint(**model_ckpt_params),
    EarlyStopping(**early_stopping_params),
    LearningRateMonitor(**lr_monitor_params)
]

# instantiate the loggers: (already done by the import)
log.info("Instantiating loggers...")


csv_logger_params = {"save_dir": "./logs/", "version": "", "name": ""}
tensorboard_logger_params = {"save_dir": "./logs/", "version": "", "name": ""}

loggers = [
    CSVLogger(**csv_logger_params),
    TensorBoardLogger(**tensorboard_logger_params),
]

# instantiate trainer:
log.info("Instantiating Trainer...")

trainer_params = {"gradient_clip_val": 200, "max_epochs": 100, "log_every_n_steps": 5}

trainer = Trainer(
    **trainer_params,
    callbacks=callbacks,
    logger=loggers
)

# Fit the trainer using the model and datamodules:
log.info("Starting training.")
trainer.fit(model=model, datamodule= datamodule)
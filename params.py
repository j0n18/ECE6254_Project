#for monitors, currently only using recon_loss

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

model_params_vae = {
    "input_dim": 2, 
    "hidden_size": 4,
    "output_dim": 6, 
    "drop_prop": 0.05
}

model_params_ae = {
    "input_dim": 2, 
    "hidden_size": 4,
    "output_dim": 6, 
    "drop_prop": 0.05
}

model_ckpt_params_vae = {
    "monitor": "recon_loss",
    "mode": "min",
    "save_top_k": 1,
    "save_last": True,
    "verbose": False,
    "dirpath": "./vae_chkpts/",
    "auto_insert_metric_name": True,
}

model_ckpt_params_ae = {
    "monitor": "recon_loss",
    "mode": "min",
    "save_top_k": 1,
    "save_last": True,
    "verbose": False,
    "dirpath": "./ae_chkpts/",
    "auto_insert_metric_name": True,
}

early_stopping_params = {
    "monitor": "recon_loss",
    "mode": "min",
    "patience": 200,
    "min_delta": 0,
}

lr_monitor_params = {"logging_interval": "epoch"}

csv_logger_params = {"save_dir": "./logs/", "version": "", "name": ""}
tensorboard_logger_params = {"save_dir": "./logs/", "version": "", "name": ""}

trainer_params = {"gradient_clip_val": 200, "max_epochs": 100, "log_every_n_steps": 5}
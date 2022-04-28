import torch
from torch import nn
import pytorch_lightning as pl
import numpy as np
from torch.distributions import Independent, Normal

from ECE6254_Project.models.modules.nets import ConvEncoder, ConvDecoder


def reparameterize(latent_mu, latent_var):
    """
    sample from latent gaussian by sampling from N(0, I) (reparameterization trick)
    B is batch size and D is latent dim
    :param latent_mu: (BxD) tensor with latent dim means
    :param latent_var: (BxD) tensor with latent dim variances
    :return: samples: (BxD) latent variable samples
    """
    eps = torch.randn_like(latent_mu)
    return latent_mu + eps * torch.exp(latent_var * 0.5)


def log_normal_pdf(sample, mean, log_var, axis=1):
    log2pi = torch.log(torch.as_tensor(2 * np.pi))
    log_var = torch.as_tensor(log_var)
    mean = torch.as_tensor(mean)
    sample = torch.as_tensor(sample)
    return torch.sum(
        -0.5 * (((sample - mean) ** 2) * torch.exp(-log_var) + log_var + log2pi),
        dim=axis
    )


class CVAE(pl.LightningModule):
    def __init__(self,
                 input_shape, hidden_layers, latent_dim,
                 mu_prior_scale=0,
                 var_prior_scale=1,
                 kl_start_epoch=50,
                 ramp_scale=4):
        super().__init__()
        self.save_hyperparameters()

        # Define the Encoder:
        self.encoder = ConvEncoder(input_shape,
                                   hidden_layers,
                                   latent_dim)

        # Define the Decoder:
        self.decoder = ConvDecoder(input_shape,
                                   hidden_layers,
                                   latent_dim,
                                   self.encoder.enc_h_out,
                                   self.encoder.enc_w_out)

        self.reconstruction_loss = nn.MSELoss(reduction='none')

    def forward(self, x):
        mu, logvar = self.encoder(x)

        # define distribution and sample z:
        normal_post = Normal(mu, torch.exp(logvar * 0.5))
        posterior = Independent(normal_post, 1)
        z = posterior.rsample()

        x_rec = self.decoder(z)
        return x_rec, z

    def compute_loss(self, batch, beta=1):
        x = batch[0]
        mu, log_var = self.encoder(x)
        z = reparameterize(mu, log_var)
        x_rec = self.decoder(z)

        # reconstruction loss = -log p(x|z)
        log_px_z = -self.reconstruction_loss(x_rec, x)
        log_px_z = torch.sum(log_px_z, dim=tuple(np.arange(0, len(log_px_z.shape))[1:]))
        recon_loss = -torch.mean(log_px_z, dim=0)

        # KL loss = log q(z|x) - log p(z)
        log_pz = log_normal_pdf(z, 0, 0)
        log_qz_x = log_normal_pdf(z, mu, log_var)
        kl_loss = torch.mean(log_qz_x - log_pz, dim=0)

        total_loss = recon_loss + beta * kl_loss

        # we are maximizing so return the negative of the ELBO estimate
        return recon_loss, kl_loss, total_loss

    def training_step(self, batch, batch_idx):
        # double check that batch_idx is correct: assumes that
        # the dataset, which sends data, inds is filling these properly

        # forward pass
        recon_loss, kl_loss, total_loss = self.compute_loss(batch)

        self.log_dict(
            {"recon_loss": recon_loss, "kl_loss": kl_loss, "total_loss": total_loss}
        )

        return total_loss

    def validation_step(self, batch, batch_idx):
        # double check that batch_idx is correct: assumes that
        # the dataset, which sends data, inds is filling these properly

        # forward pass
        recon_loss, kl_loss, total_loss = self.compute_loss(batch)

        self.log_dict(
            {"recon_loss": recon_loss, "kl_loss": kl_loss, "total_loss": total_loss}
        )

        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        return {"optimizer": optimizer,
                "monitor": "total_loss"}

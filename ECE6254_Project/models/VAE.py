import torch
from torch import nn
import pytorch_lightning as pl
from torch.distributions import Independent, Normal
from torch.distributions.kl import kl_divergence

from ECE6254_Project.models.modules.nets import MLP_Encoder, MLP_Decoder

class VAE(pl.LightningModule):
    def __init__(self,
        input_dim, 
        hidden_size,
        output_dim, 
        drop_prop,
        mu_prior_scale = 0,
        var_prior_scale = 0.1,
        kl_start_epoch = 50,
        ramp_scale = 4):

        super().__init__()
        self.save_hyperparameters()

        #Define the Encoder:
        self.encoder = MLP_Encoder(input_dim,
                                hidden_size,
                                output_dim, 
                                drop_prop)

        #Define the Decoder:
        self.decoder = MLP_Decoder(output_dim,
                                hidden_size,
                                input_dim, 
                                drop_prop)



    def forward(self, x):

        mu, logvar = self.encoder(x)

        #define distribution and sample z:
        normal_post = Normal(mu, torch.exp(logvar * 0.5))
        posterior = Independent(normal_post, 1)
        z = posterior.rsample()
        
        x_rec = self.decoder(z)
        return x_rec, z


    def compute_loss(self, batch):

        hps = self.hparams

        #build tensor dataset like this (with this ordering):
        data, _ = batch

        mu, logvar = self.encoder(data)
        normal_post = Normal(mu, torch.exp(logvar * 0.5))
        posterior = Independent(normal_post, 1)

        z = posterior.rsample()

        data_rec = self.decoder(z)

        #Reconstruction cost:
        recon_loss_fcn = nn.MSELoss(reduction = "none")
        recon_loss = recon_loss_fcn(data, data_rec)
        recon_loss = torch.mean(recon_loss) #need scalar value for .backward()

        #KL cost:
        #compute KL divergence between latent dist and normal dist:
        normal_post = Normal(mu, torch.exp(logvar * 0.5))
        posterior = Independent(normal_post, 1)

        mu_prior = torch.zeros_like(mu)
        var_prior = hps.var_prior_scale * torch.ones_like(logvar)

        normal_prior = Normal(mu_prior, var_prior)
        prior = Independent(normal_prior, 1)

        kl_loss = kl_divergence(prior, posterior)
        kl_loss = torch.mean(kl_loss)

        if self.current_epoch <= hps.kl_start_epoch:
            kl_ramp = 0
        else:
            kl_ramp = (self.current_epoch - hps.kl_start_epoch) / (hps.ramp_scale * hps.kl_start_epoch + 1)

        #low norm mean cost:
        #mu_norm_loss = torch.norm(mu, p = 2)

        #similarity matching cost:

        #################
        ## ** TODO ** ###
        #################

        # Final loss #make sure kl divergence is non-negative
        total_loss = recon_loss + kl_ramp * kl_loss #+ mu_norm_loss

        return recon_loss, kl_loss, total_loss #mu_norm_loss, total_loss


    def training_step(self, batch, batch_idx):
        #double check that batch_idx is correct: assumes that
        #the dataset, which sends data, inds is filling these properly

        # forward pass
        recon_loss, kl_loss, total_loss = self.compute_loss(batch)

        self.log_dict(
            {"recon_loss": recon_loss, "kl_loss": kl_loss, "total_loss": total_loss}
        )

        return total_loss


    def validation_step(self, batch, batch_idx):
        #double check that batch_idx is correct: assumes that
        #the dataset, which sends data, inds is filling these properly

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
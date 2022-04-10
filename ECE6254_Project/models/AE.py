import torch
from torch import nn
import pytorch_lightning as pl
from ECE6254_Project.models.modules.nets import MLP_Encoder, MLP_Decoder

class AE(pl.LightningModule):
    def __init__(self,
        input_dim, 
        hidden_size,
        output_dim, 
        drop_prop,
        beta = 0.01,
        ):

        super().__init__()
        self.save_hyperparameters()

        #Define the Encoder:
        self.encoder = MLP_Encoder(input_dim,
                                hidden_size,
                                output_dim, 
                                drop_prop,
                                AE = True)

        #Define the Decoder:
        self.decoder = MLP_Decoder(output_dim,
                                hidden_size,
                                input_dim, 
                                drop_prop)



    def forward(self, x):

        z = self.encoder(x)
        x_rec = self.decoder(z)

        return x_rec, z


    def compute_loss(self, batch):

        #build tensor dataset like this (with this ordering):
        data, _ = batch

        z = self.encoder(data)

        data_rec = self.decoder(z)

        #Reconstruction cost:
        recon_loss_fcn = nn.MSELoss(reduction = "none")
        recon_loss = recon_loss_fcn(data, data_rec)
        recon_loss = torch.mean(recon_loss) #need scalar value for .backward()

        #low norm mean cost:
        z_norm_loss = self.hparams.beta * torch.norm(z, p = 2)

        #similarity matching cost:

        #################
        ## ** TODO ** ###
        #################

        total_loss = recon_loss + z_norm_loss

        return recon_loss, z_norm_loss, total_loss


    def training_step(self, batch, batch_idx):
        #double check that batch_idx is correct: assumes that
        #the dataset, which sends data, inds is filling these properly

        # forward pass
        recon_loss, z_norm_loss, total_loss = self.compute_loss(batch)

        self.log_dict(
            {"recon_loss": recon_loss, "total_loss": total_loss}
        )

        return recon_loss


    def validation_step(self, batch, batch_idx):
        #double check that batch_idx is correct: assumes that
        #the dataset, which sends data, inds is filling these properly

        # forward pass
        recon_loss, z_norm_loss, total_loss = self.compute_loss(batch)

        self.log_dict(
            {"recon_loss": recon_loss, "total_loss": total_loss}
        )

        return recon_loss


    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        return {"optimizer": optimizer,
                "monitor": "recon_loss"}
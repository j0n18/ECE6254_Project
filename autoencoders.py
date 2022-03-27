import numpy as np
import torch
from torch import nn
from torch.optim import Adam


def log_normal_pdf(sample, mean, log_var, axis=1):
    log2pi = torch.log(torch.as_tensor(2 * np.pi))
    log_var = torch.as_tensor(log_var)
    mean = torch.as_tensor(mean)
    sample = torch.as_tensor(sample)
    return torch.sum(
        -0.5 * ((sample - mean) ** 2 * torch.exp(-log_var) + log_var + log2pi),
        dim=axis
    )


class Reshape(nn.Module):

    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class ConvolutionalVAE(nn.Module):

    def __init__(self, input_shape, hidden_layers, latent_dim):
        super(ConvolutionalVAE, self).__init__()

        """ BUILD THE ENCODER """
        self.flatten = nn.Flatten(start_dim=1)
        self.latent_dim = latent_dim

        channels_in, h, w = input_shape
        enc_layers = []
        layer_dims = [channels_in] + hidden_layers
        stride = (2, 2)
        kernel_sz = (3, 3)
        padding = (1, 1)
        final_h, final_w = h, w
        for h in range(1, len(layer_dims)):
            enc_layers.append(nn.Sequential(
                nn.Conv2d(layer_dims[h - 1], out_channels=layer_dims[h],
                          kernel_size=kernel_sz,
                          stride=stride,
                          padding=padding),
                nn.BatchNorm2d(layer_dims[h]),
                nn.LeakyReLU()
            ))
            final_h = int(np.floor((final_h + 2 * padding[0] - (kernel_sz[0] - 1) - 1) / stride[0] + 1))
            final_w = int(np.floor((final_w + 2 * padding[1] - (kernel_sz[1] - 1) - 1) / stride[1] + 1))

        # mapping to the latents
        enc_layers.append(nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=layer_dims[-1] * final_h * final_w, out_features=2 * latent_dim)
        ))

        self.encoder = nn.Sequential(*enc_layers)

        """ BUILD THE DECODER """
        self.decoder_input = nn.Sequential(
            nn.Linear(latent_dim, layer_dims[-1] * final_h * final_w),
            nn.ReLU()
        )
        self.decoder_view = Reshape(shape=(-1, layer_dims[-1], final_h, final_w))
        dec_layers = []
        output_padding = (1, 1)
        hidden_layers.reverse()
        for h in range(0, len(hidden_layers) - 1):
            dec_layers.append(nn.Sequential(
                nn.ConvTranspose2d(hidden_layers[h], out_channels=hidden_layers[h + 1],
                                   kernel_size=kernel_sz,
                                   stride=stride,
                                   padding=padding,
                                   output_padding=output_padding),
                nn.BatchNorm2d(hidden_layers[h + 1]),
                nn.LeakyReLU()
            ))
            final_h = (final_h - 1) * stride[0] - 2 * padding[0] + (kernel_sz[0] - 1) + output_padding[0] + 1
            final_w = (final_w - 1) * stride[1] - 2 * padding[1] + (kernel_sz[1] - 1) + output_padding[1] + 1

            # TODO: figure out how to reverse the output shape more generally (how much does stride > 1 matter)
            if final_h == 4:
                output_padding = (0, 0)
            else:
                output_padding = (1, 1)

        # final output
        dec_layers.append(nn.Sequential(
            nn.ConvTranspose2d(hidden_layers[-1], channels_in,
                               kernel_size=kernel_sz,
                               stride=(2, 2),
                               padding=padding,
                               output_padding=output_padding),
            nn.Tanh()))

        self.decoder = nn.Sequential(*dec_layers)

        """ SET THE OPTIMIZER """
        self.parameters = list(self.encoder.parameters()) + \
                          list(self.decoder_input.parameters()) + \
                          list(self.decoder.parameters())

        # default optimizer
        self.optimizer = Adam(self.parameters, lr=0.001)

        """ DEFINE THE LOSSES """
        self.reconstruction_loss = nn.MSELoss(reduction='none')

    def encode(self, x):
        """
        return distribution parameters over latent variables
        B is batch size, C is no. of channels, H and W are image height and width
        :param x: (BxCxHxW) tensor with input data
        :return: mu: (BxD) tensor with means over latent variables
        :return: var: (BxD) tensor with variances over latent variables
        """
        mu, log_var = torch.split(self.encoder(x), split_size_or_sections=2, dim=1)
        return mu, log_var

    def decode(self, z):
        """
        map the latent variable z to an element x in the input space
        :param z: (BxD) latent variable
        :return: x: (BxCxHxW) element in the input space
        """
        x_flat = self.decoder_input(z)
        x_reshaped = self.decoder_view(x_flat)
        x = self.decoder(x_reshaped)

        return x

    def reparameterize(self, latent_mu, latent_var):
        """
        sample from latent gaussian by sampling from N(0, I) (reparameterization trick)
        B is batch size and D is latent dim
        :param latent_mu: (BxD) tensor with latent dim means
        :param latent_var: (BxD) tensor with latent dim variances
        :return: samples: (BxD) latent variable samples
        """
        eps = torch.randn_like(latent_mu)
        return latent_mu + eps * torch.exp(latent_var * 0.5)

    def forward(self, x):
        """
        return the reconstructed version of x
        :param x: (BxCxHxW) input image
        :return: x_rec: (BxCxHxW) reconstructed image
        :return: mu: (BxD) the mean vector of latents
        :return: log_var: (BxD) vector of diagonal log_vars for latents
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(latent_mu=mu, latent_var=log_var)
        x_rec = self.decode(z)
        return x_rec, mu, log_var

    def loss_fun(self, x):
        """
        computes the monte carlo estimate of the ELBO loss for one input x
        :param x: (BxCxHxW) input image(s)
        :return: loss: 1d tensor containing the loss estimate
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_rec = self.decode(z)

        # reconstruction loss = log p(x|z)
        recon_loss = self.reconstruction_loss(x_rec, x)
        recon_loss = -torch.sum(recon_loss, dim=(1, 2, 3))

        # KL loss = log p(z) - log q(z|x)
        log_pz = log_normal_pdf(z, 0, 0)
        log_qz_x = log_normal_pdf(z, mu, log_var)

        # we are maximizing so return the negative of the ELBO estimate
        return -torch.mean(recon_loss + log_pz - log_qz_x, dim=0)

    def training_step(self, x):
        """
        performs one training step using the input x
        :param x: (BxCxHxW) input batch of images
        :return loss: the loss after the training the step
        """
        # zero gradients
        self.optimizer.zero_grad()

        # forward pass
        loss = self.loss_fun(x)

        # backward pass
        loss.backward()

        # apply gradients
        self.optimizer.step()

        return loss

    def sample(self):
        # TODO
        pass

    def generate(self):
        # TODO
        pass

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_recons_loss(self, loss):
        self.reconstruction_loss = loss

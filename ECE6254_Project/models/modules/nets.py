import torch
from torch import nn
import numpy as np


class Reshape(nn.Module):

    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class MLP_Encoder(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_size,
                 output_dim,
                 drop_prop,
                 AE=False):

        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.drop_prop = drop_prop
        self.AE = AE

        # self.dropout = nn.Dropout(p = self.drop_prop)
        self.linear1 = nn.Linear(input_dim, hidden_size)
        # self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU(inplace=True)

        if self.AE:
            self.linear2 = nn.Linear(hidden_size, output_dim, bias=True)
        else:
            self.linear2 = nn.Linear(hidden_size, 2 * output_dim, bias=True)

    def forward(self, x):

        # x = self.dropout(x)
        x = self.linear1(x)
        # x = self.bn1(x)
        x = self.relu(x)
        x = self.linear2(x)
        # x = self.dropout(x)

        if self.AE:
            return x

        mu, logvar = torch.split(x, self.output_dim, dim=1)
        return mu, logvar


class MLP_Decoder(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_size,
                 output_dim,
                 drop_prop):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.drop_prop = drop_prop

        # self.dropout = nn.Dropout(p = self.drop_prop)
        self.linear1 = nn.Linear(input_dim, hidden_size, bias=True)
        # self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        # x = self.dropout(x)
        x = self.linear1(x)
        # x = self.bn1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class ConvEncoder(nn.Module):

    def __init__(self, input_shape, hidden_layers, latent_dim,
                 stride=(2, 2), kernel_sz=(3, 3), padding=(1, 1), AE=False):
        super().__init__()

        self.flatten = nn.Flatten(start_dim=1)
        self.latent_dim = latent_dim
        self.AE = AE
        channels_in, h, w = input_shape
        enc_layers = []
        layer_dims = [channels_in] + hidden_layers
        enc_h_out, enc_w_out = [h], [w]
        for h in range(1, len(layer_dims)):
            enc_layers.append(nn.Sequential(
                nn.Conv2d(layer_dims[h - 1], out_channels=layer_dims[h],
                          kernel_size=kernel_sz,
                          stride=stride,
                          padding=padding),
                nn.BatchNorm2d(layer_dims[h]),
                nn.LeakyReLU()
            ))
            enc_h_out.append(
                int(np.floor((enc_h_out[h - 1] + 2 * padding[0] - (kernel_sz[0] - 1) - 1) / stride[0] + 1)))
            enc_w_out.append(
                int(np.floor((enc_w_out[h - 1] + 2 * padding[1] - (kernel_sz[1] - 1) - 1) / stride[1] + 1)))

        final_h, final_w = enc_h_out[-1], enc_w_out[-1]
        # mapping to the latents
        if self.AE:
            enc_layers.append(nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features=layer_dims[-1] * final_h * final_w, out_features=latent_dim)
            ))
        else:
            enc_layers.append(nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features=layer_dims[-1] * final_h * final_w, out_features=2 * latent_dim)
            ))

        self.encoder = nn.Sequential(*enc_layers)
        self.enc_h_out = enc_h_out
        self.enc_w_out = enc_w_out

    def forward(self, x):
        """
        return distribution parameters over latent variables
        B is batch size, C is no. of channels, H and W are image height and width
        :param x: (BxCxHxW) tensor with input data
        :return: mu: (BxD) tensor with means over latent variables
        :return: var: (BxD) tensor with variances over latent variables
        """
        if self.AE:
            return self.encoder(x)

        mu, log_var = torch.split(self.encoder(x), split_size_or_sections=self.latent_dim, dim=1)
        return mu, log_var


class ConvDecoder(nn.Module):

    def __init__(self, input_shape, hidden_layers, latent_dim,
                 enc_h_out, enc_w_out,
                 stride=(2, 2), kernel_sz=(3, 3), padding=(1, 1)):
        super().__init__()

        self.latent_dim = latent_dim

        channels_in, h, w = input_shape
        layer_dims = [channels_in] + hidden_layers
        final_h, final_w = enc_h_out[-1], enc_w_out[-1]

        self.decoder_input = nn.Sequential(
            nn.Linear(latent_dim, layer_dims[-1] * final_h * final_w),
            nn.ReLU()
        )
        self.decoder_view = Reshape(shape=(-1, layer_dims[-1], final_h, final_w))
        dec_layers = []
        enc_h_out.reverse()
        enc_w_out.reverse()
        hidden_layers.reverse()
        dec_h_out, dec_w_out = [final_h], [final_w]
        for h in range(0, len(hidden_layers)):
            # adjust the output padding to get the same output shape
            if enc_h_out[h + 1] % 2 == 0:
                output_padding = (1, 1)
            else:
                output_padding = (0, 0)

            if h != len(hidden_layers) - 1:
                dec_layers.append(nn.Sequential(
                    nn.ConvTranspose2d(hidden_layers[h], out_channels=hidden_layers[h + 1],
                                       kernel_size=kernel_sz,
                                       stride=stride,
                                       padding=padding,
                                       output_padding=output_padding),
                    nn.BatchNorm2d(hidden_layers[h + 1]),
                    nn.LeakyReLU()
                ))
            else:
                # final output
                dec_layers.append(nn.Sequential(
                    nn.ConvTranspose2d(hidden_layers[h], out_channels=channels_in,
                                       kernel_size=kernel_sz,
                                       stride=stride,
                                       padding=padding,
                                       output_padding=output_padding),
                    nn.Tanh()))

            dec_h_out.append(
                (dec_h_out[h] - 1) * stride[0] - 2 * padding[0] + (kernel_sz[0] - 1) + output_padding[0] + 1)
            dec_w_out.append(
                (dec_w_out[h] - 1) * stride[1] - 2 * padding[1] + (kernel_sz[1] - 1) + output_padding[1] + 1)

        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, z):
        """
        map the latent variable z to an element x in the input space
        :param z: (BxD) latent variable
        :return: x: (BxCxHxW) element in the input space
        """
        x_flat = self.decoder_input(z)
        x_reshaped = self.decoder_view(x_flat)
        x = self.decoder(x_reshaped)

        return x

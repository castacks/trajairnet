"""Modified from https://github.com/timbmg/VAE-CVAE-MNIST"""

import torch
import torch.nn as nn

# from utils import idx2onehot


class CVAE(nn.Module):

    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes,
                 conditional=True, num_labels=0):

        super().__init__()

        if conditional:
            assert num_labels > 0

        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list

        self.latent_size = latent_size

        self.encoder = Encoder(
            encoder_layer_sizes, latent_size, conditional, num_labels)
        self.decoder = Decoder(
            decoder_layer_sizes, latent_size, conditional, num_labels)
        self.linear = nn.Linear(latent_size,num_labels)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x, c=None):

        # if x.dim() > 2:
        #     x = x.view(-1, 28*28)

        means, log_var = self.encoder(x, c)
        z = self.reparameterize(means, log_var)
        # b_z = self.softmax(self.linear(z))
        # cz = b_z*c
#         print("cvae",c.shape,z.shape)
        recon_x = self.decoder(c, z)

        return recon_x, means, log_var, z

    def reparameterize(self, mu, log_var):

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return mu + eps * std

    def inference(self, z, c=None):
        # b_z = self.softmax(self.linear(z))
        # cz = b_z*c
        # print("")
#         print("cvae",c.shape,z.shape)
        recon_x = self.decoder(c, z)

        return recon_x


class Encoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, num_labels):

        super().__init__()

        self.conditional = conditional
        if self.conditional:
            layer_sizes[0] += num_labels

        self.MLP = nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            # print(in_size,out_size)
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)

    def forward(self, x, c=None):

        if self.conditional:
            x = torch.cat((x, c), dim=-1)

        x = self.MLP(x)

        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars


class Decoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, num_labels):

        super().__init__()

        self.MLP = nn.Sequential()

        self.conditional = conditional
        if self.conditional:
            input_size = num_labels + latent_size
        else:
            input_size = latent_size

        for i, (in_size, out_size) in enumerate(zip([input_size]+layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i+1 < len(layer_sizes):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.Tanh())
            else:
                self.MLP.add_module(name="sigmoid", module=nn.Tanh())

                

    def forward(self, c, z):

        if self.conditional:
            z = torch.cat((z, c), dim=-1)
            # print(z)

        x = self.MLP(z)
        # print("x",x)

        return x

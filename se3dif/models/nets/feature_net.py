## Code based on https://github.com/facebookresearch/DeepSDF
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
    def forward(self, x):
        x_proj = torch.einsum('...,b->...b',x, self.W)* 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

## Activation Function
class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


## Time and Latent Code conditioned Feature encoder ##
class TimeLatentFeatureEncoder(nn.Module):
    def __init__(
        self,
        latent_size,
        dims,
        in_dim = 3,
        enc_dim = 256,
        out_dim = 1,
        dropout=None,
        dropout_prob=0.0,
        norm_layers=(),
        latent_in=(),
        weight_norm=False,
        xyz_in_all=None,
        use_tanh=False,
        latent_dropout=False,
        feats_layers = None
    ):
        super(TimeLatentFeatureEncoder, self).__init__()

        ## Time Embedings Encoder ##
        self.time_embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=enc_dim),
            nn.Linear(enc_dim, enc_dim),
            nn.SiLU(),
        )
        self.x_embed = nn.Sequential(
            nn.Linear(in_dim, enc_dim),
            nn.SiLU(),
        )

        self.out_dim = out_dim
        self.latent_size = latent_size
        self.in_dim = in_dim

        def make_sequence():
            return []

        dims = [latent_size + enc_dim + in_dim] + dims + [out_dim]

        self.num_layers = len(dims)
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.latent_dropout = latent_dropout
        if self.latent_dropout:
            self.lat_dp = nn.Dropout(0.2)

        self.xyz_in_all = xyz_in_all
        self.weight_norm = weight_norm

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in latent_in:
                out_dim = dims[layer + 1] - dims[0]
            else:
                out_dim = dims[layer + 1]
                if self.xyz_in_all and layer != self.num_layers - 2:
                    out_dim -= 3

            if weight_norm and layer in self.norm_layers:
                setattr(
                    self,
                    "lin" + str(layer),
                    nn.utils.weight_norm(nn.Linear(dims[layer], out_dim)),
                )
            else:
                setattr(self, "lin" + str(layer), nn.Linear(dims[layer], out_dim))

            if (
                (not weight_norm)
                and self.norm_layers is not None
                and layer in self.norm_layers
            ):
                setattr(self, "bn" + str(layer), nn.LayerNorm(out_dim))

        self.use_tanh = use_tanh
        if use_tanh:
            self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.th = nn.Tanh()

        if feats_layers is None:
            self.feats_layers = list(np.arange(0,self.num_layers-1))
        else:
            self.feats_layers = feats_layers

    # input: N x (L+3)
    def forward(self, input, timesteps, latent_vecs=None):

        ## Encode TimeStep
        t_emb = self.time_embed(timesteps)
        ## Encode Position
        x_emb = self.x_embed(input)
        xyz = x_emb + t_emb

        if (latent_vecs is not None):
            latent_vecs = F.dropout(latent_vecs, p=0.2, training=self.training)
            x = torch.cat([latent_vecs, xyz, input], -1)
        else:
            x = torch.cat([xyz, input], -1)
        x0 = x.clone()

        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))
            if layer in self.latent_in:
                x = torch.cat([x, x0], -1)
            elif layer != 0 and self.xyz_in_all:
                x = torch.cat([x, input], -1)
            x = lin(x)
            # last layer Tanh
            if layer == self.num_layers - 2 and self.use_tanh:
                x = self.tanh(x)
            if layer < self.num_layers - 2:
                if (
                    self.norm_layers is not None
                    and layer in self.norm_layers
                    and not self.weight_norm
                ):
                    bn = getattr(self, "bn" + str(layer))
                    x = bn(x)
                x = self.relu(x)
                if self.dropout is not None and layer in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

        if hasattr(self, "th"):
            x = self.th(x)

        return x


if __name__ == '__main__':

    x_dim = 3
    lat_dim = 128
    def eval(model):
        x_in = torch.randn((10, 1500, x_dim))
        t_in = torch.rand_like(x_in[...,0])
        lat_in = torch.randn((10, 1500, lat_dim))
        z_out = model(x_in, t_in, lat_in)
        print(z_out.shape)

    model = TimeLatentFeatureEncoder(
        enc_dim=128,
        latent_size= 128,
        dims = [512,512],
        out_dim=2
    )
    eval(model)



import torch
import numpy as np
import torch.nn as nn

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

class NaiveSE3DiffusionModel(nn.Module):
    def __init__(self, energy=False):
        super().__init__()

        input_size = 12
        enc_dim = 128
        if energy:
            output_size = 1
        else:
            output_size = 6

        self.network = nn.Sequential(
            nn.Linear(2*enc_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_size)
        )

        ## Time Embedings Encoder ##
        self.time_embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=enc_dim),
            nn.Linear(enc_dim, enc_dim),
            nn.SiLU(),
        )
        self.x_embed = nn.Sequential(
            nn.Linear(input_size, enc_dim),
            nn.SiLU(),
        )

    def marginal_prob_std(self, t, sigma=0.5):
        return torch.sqrt((sigma ** (2 * t) - 1.) / (2. * np.log(sigma)))

    def forward(self, x, R, k):
        std = self.marginal_prob_std(k)
        x_R_input = torch.cat((x, R.reshape(R.shape[0], -1)), dim=-1)
        z = self.x_embed(x_R_input)
        z_time = self.time_embed(k)
        z_in = torch.cat((z, z_time),dim=-1)
        v = self.network(z_in)
        return v/(std[:,None].pow(2))


class GraspDiffusionFields(nn.Module):
    ''' Grasp DiffusionFields. SE(3) diffusion model to learn 6D grasp distributions. See
        SE(3)-DiffusionFields: Learning cost functions for joint grasp and motion optimization through diffusion
    '''
    def __init__(self, vision_encoder, geometry_encoder, points, feature_encoder, decoder):
        super().__init__()
        ## Register points to map H to points ##
        self.register_buffer('points', points)
        ## Vision Encoder. Map observation to visual latent code ##
        self.vision_encoder = vision_encoder
        ## vision latent code
        self.z = None
        ## Geometry Encoder. Map H to points ##
        self.geometry_encoder = geometry_encoder
        ## Feature Encoder. Get SDF and latent features ##
        self.feature_encoder = feature_encoder
        ## Decoder ##
        self.decoder = decoder

    def set_latent(self, O, batch = 1):
        self.z = self.vision_encoder(O.squeeze(1))
        self.z = self.z.unsqueeze(1).repeat(1, batch, 1).reshape(-1, self.z.shape[-1])

    def forward(self, H, k):
        ## 1. Represent H with points
        p = self.geometry_encoder(H, self.points)
        k_ext = k.unsqueeze(1).repeat(1, p.shape[1])
        z_ext = self.z.unsqueeze(1).repeat(1, p.shape[1], 1)
        ## 2. Get Features
        psi = self.feature_encoder(p, k_ext, z_ext)
        ## 3. Flat and get energy
        psi_flatten = psi.reshape(psi.shape[0], -1)
        e = self.decoder(psi_flatten)
        return e

    def compute_sdf(self, x):
        k = torch.rand_like(x[..., 0])
        psi = self.feature_encoder(x, k, self.z)
        return psi[..., 0]

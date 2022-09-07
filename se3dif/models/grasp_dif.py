import torch
import torch.nn as nn



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

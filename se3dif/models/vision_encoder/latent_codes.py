import torch
import torch.nn as nn
import math

class LatentCodes(nn.Module):
    def __init__(
        self,
        num_scenes,
        latent_size,
        code_bound=1.,
        std = 1.0
    ):
        super(LatentCodes, self).__init__()

        self.lat_vecs = torch.nn.Embedding(num_scenes, latent_size, max_norm=code_bound)
        torch.nn.init.normal_(
            self.lat_vecs.weight.data,
            0.0,
            std / math.sqrt(latent_size),
        )
        print('latent codes set')

    def forward(self, idxs):
        lat_vecs = self.lat_vecs(idxs.int())
        return lat_vecs


if __name__ == '__main__':

    num_scenes  = 10
    latent_size = 256
    code_bound = 1.

    def eval(model):
        id = torch.randint(low=0, high=num_scenes, size=[100])
        out = model(id)
        print(out.shape)

    model = LatentCodes(num_scenes, latent_size, code_bound)
    eval(model)
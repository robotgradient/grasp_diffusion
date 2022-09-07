import torch
import torch.nn as nn


def map_projected_points(H, p):
    p_ext = torch.cat((p, torch.ones_like(p[..., :1])), -1)
    p_alig = torch.einsum('...md,pd->...pm', H, p_ext)[..., :-1]
    return p_alig
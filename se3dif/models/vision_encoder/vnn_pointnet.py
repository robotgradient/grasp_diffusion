## This file is based on https://github.com/FlyingGiraffe/vnn
import torch
import torch.nn as nn
from .equiv_layers import VNLinearLeakyReLU, VNLinear, VNResnetBlockFC, VNLeakyReLU, VNStdFeature, get_graph_feature_cross


def maxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out

def meanpool(x, dim=-1, keepdim=False):
    out = x.mean(dim=dim, keepdim=keepdim)
    return out


class VNNPointnet2(nn.Module):
    ''' PointNet-2 Net '''
    def __init__(self, pointnet_radius=0.1, hidden_dim=512, in_features=3,
                 out_features = 512, device='cpu'):
        super().__init__()
        if out_features%3 !=0:
            print('This might break. The module expects a feature to be a multiple of 3.')

        self.vnn_resnet = VNN_ResnetPointnet(c_dim=int(out_features/3), device=device)

    def forward(self, pc):

        xyz_feats = self.vnn_resnet(pc)
        return xyz_feats.reshape(pc.shape[0], -1)


class VNN_ResnetPointnet(nn.Module):
    ''' DGCNN-based VNN vision_encoder network with ResNet blocks.

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''

    def __init__(self, c_dim=128, dim=3, hidden_dim=128, k=20, meta_output=None, device = 'cpu'):
        super().__init__()
        self.c_dim = c_dim
        self.k = k
        self.meta_output = meta_output

        self.conv_pos = VNLinearLeakyReLU(3, 128, negative_slope=0.0, share_nonlinearity=False, use_batchnorm=False)
        self.fc_pos = VNLinear(128, 2 * hidden_dim)
        self.block_0 = VNResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_1 = VNResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_2 = VNResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_3 = VNResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_4 = VNResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.fc_c = VNLinear(hidden_dim, c_dim)

        self.actvn_c = VNLeakyReLU(hidden_dim, negative_slope=0.0, share_nonlinearity=False)
        self.pool = meanpool

        if meta_output == 'invariant_latent':
            self.std_feature = VNStdFeature(c_dim, dim=3, normalize_frame=True, use_batchnorm=False)
        elif meta_output == 'invariant_latent_linear':
            self.std_feature = VNStdFeature(c_dim, dim=3, normalize_frame=True, use_batchnorm=False)
            self.vn_inv = VNLinear(c_dim, 3)
        elif meta_output == 'equivariant_latent_linear':
            self.vn_inv = VNLinear(c_dim, 3)

        self.device = device

    def forward(self, p):
        batch_size = p.size(0)
        p = p.unsqueeze(1).transpose(2, 3)
        # mean = get_graph_mean(p, k=self.k)
        # mean = p_trans.mean(dim=-1, keepdim=True).expand(p_trans.size())
        feat = get_graph_feature_cross(p, k=self.k, device=self.device)
        net = self.conv_pos(feat)
        net = self.pool(net, dim=-1)

        net = self.fc_pos(net)

        net = self.block_0(net)
        pooled = self.pool(net, dim=-1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)

        net = self.block_1(net)
        pooled = self.pool(net, dim=-1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)

        net = self.block_2(net)
        pooled = self.pool(net, dim=-1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)

        net = self.block_3(net)
        pooled = self.pool(net, dim=-1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)

        net = self.block_4(net)

        # Recude to  B x F
        net = self.pool(net, dim=-1)

        c = self.fc_c(self.actvn_c(net))

        if self.meta_output == 'invariant_latent':
            c_std, z0 = self.std_feature(c)
            return c, c_std
        elif self.meta_output == 'invariant_latent_linear':
            c_std, z0 = self.std_feature(c)
            c_std = self.vn_inv(c_std)
            return c, c_std
        elif self.meta_output == 'equivariant_latent_linear':
            c_std = self.vn_inv(c)
            return c, c_std

        return c



if __name__ == '__main__':

    device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')

    def eval(model):
        x_in = torch.randn((2, 1024, 3)).to(device)
        z = model(x_in)
        print(z.shape)


    ## RESNET POINTNET
    model = VNNPointnet2(out_features=126, device=device).to(device)
    eval(model)

import torch
import torch.nn as nn

class SDFLoss():
    def __init__(self, field='sdf', delta = 0.6, grad=True):
        self.field = field
        self.delta = delta

        self.grad = grad

    def loss_fn(self, model, model_input, ground_truth, val=False):
        loss_dict = dict()
        label = ground_truth[self.field].squeeze().reshape(-1)

        ## Set input ##
        x_sdf = model_input['x_sdf'].detach().requires_grad_()
        c = model_input['visual_context']

        ## Compute model output ##
        model.set_latent(c, batch=x_sdf.shape[1])
        sdf = model.compute_sdf(x_sdf.view(-1, 3))

        ## Reconstruction Loss ##
        loss = nn.L1Loss(reduction='mean')
        pred_clip_sdf = torch.clip(sdf, -10., self.delta)
        target_clip_sdf = torch.clip(label, -10., self.delta)
        l_rec = loss(pred_clip_sdf, target_clip_sdf)

        ## Total Loss
        loss_dict[self.field] = l_rec

        info = {'sdf': sdf}
        return loss_dict, info


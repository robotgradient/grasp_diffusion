import torch
import numpy as np

import theseus as th
from theseus.geometry.so3 import SO3
from torch.autograd import grad


B = 10

def get_sample_from_data(B=100):
    x_data = torch.Tensor([[0.3, 0.3, 0.3],
                           [-0.5, 1.2, -0.7]])
    theta_data = torch.Tensor([[0., 0.0, 0.0],
                               [-0.3, 1.2, -0.4]])
    R_data = SO3().exp_map(theta_data).to_matrix()
    idx = np.random.randint(0, 2, (B,))
    _x = x_data[idx, :]
    _R = R_data[idx, ...]
    return _x, _R

def sample_from_se3_gaussian(x_tar, R_tar, std):
    x_eps = std[:,None]*torch.rand_like(x_tar)
    theta_eps = std[:,None]*torch.rand_like(x_tar)
    rot_eps = SO3().exp_map(theta_eps).to_matrix()

    _x = x_tar + x_eps
    _R = torch.einsum('bmn,bnk->bmk',R_tar, rot_eps)
    return _x, _R

def se3_log_probability_normal(x, R, x_tar, R_tar, std):

    ## Send to Theseus ##
    _R_tar = SO3()
    _R_tar.update(R_tar)

    if type(R) == torch.Tensor:
        _R = SO3()
        _R.update(R)
        R = _R


    ## Compute distance in R^3 + SO(3) ##
    R_tar_inv = _R_tar.inverse()
    dR = th.compose(R_tar_inv, R)
    dtheta = dR.log_map()

    dx = (x - x_tar)

    dist = torch.cat((dx, dtheta), dim=-1)
    return -.5*dist.pow(2).sum(-1)/(std.pow(2))

def se3_score_normal(x, R, x_tar, R_tar, std):
    if type(R) == torch.Tensor:
        _R = SO3()
        _R.update(R)
        R = _R


    theta = R.log_map()
    x_theta = torch.cat((x, theta), dim=-1)
    x_theta.requires_grad_(True)
    x = x_theta[..., :3]
    R = SO3.exp_map(x_theta[..., 3:])
    d = se3_log_probability_normal(x, R, x_tar, R_tar, std)
    v = grad(d.sum(), x_theta, only_inputs=True)[0]
    return v

def step(x, R, v):
    rot = SO3.exp_map(v[..., 3:]).to_matrix()
    R_1 = torch.einsum('bmn,bnk->bmk', rot, R)

    x_1 = x + v[...,:3]
    return x_1, R_1


########## Evaluation  #########

TEST_SE3 = True
if TEST_SE3:

    R_tar = SO3.rand(B).to_matrix()
    x_tar = torch.randn(B, 3)

    H = torch.eye(4)[None,...].repeat(B, 1, 1)
    H[:, :3, :3] = R_tar
    H[:, :3, -1] = x_tar
    from se3dif.visualization.grasp_visualization import visualize_grasps
    colors = torch.zeros_like(x_tar)
    #colors[:,0] = torch.ones_like(colors[:,0])
    colors[0,0] = 1
    scene = visualize_grasps(Hs=H, colors=colors)
    #scene.show()

    R0 = SO3.rand(B).to_matrix()
    x0 = torch.randn(B, 3)

    log_prob = se3_log_probability_normal(x0, R0, x_tar, R_tar, torch.ones_like(x_tar[:,0]))
    prob = torch.exp(log_prob)
    c_x = (prob - prob.min()) / (prob.max() + prob.min())
    colors[:, 0] = c_x

    T = 5000
    dt= 0.01
    trj = torch.zeros(0)
    for t in range(T):
        d = se3_log_probability_normal(x0, R0, x_tar=x_tar, R_tar=R_tar, std=torch.ones(x_tar.shape[0]))
        trj = torch.cat((trj, d.sum()[None]), )
        print(d.sum())
        v = se3_score_normal(x0, R0, x_tar=x_tar, R_tar=R_tar, std=torch.ones(x_tar.shape[0]))
        _s = v*dt
        x0, R0 = step(x0, R0, _s)


    import matplotlib.pyplot as plt
    trj_np = trj.numpy()
    plt.plot(trj)
    plt.show()


########## Build a SE(3) Diffusion Model ###########

def marginal_prob_std(t, sigma=0.5):
    return torch.sqrt((sigma ** (2 * t) - 1.) / (2. * np.log(sigma)))

from se3dif.models.grasp_dif import NaiveSE3DiffusionModel

energy = False
model = NaiveSE3DiffusionModel(energy=energy)


from torch import optim
optimizer = optim.AdamW(model.parameters(), lr=0.0005)


K = 1000
B = 500
for k in range(K):
    t = torch.rand(B) + 10e-3
    std = marginal_prob_std(t)
    x, R = get_sample_from_data(B)

    x_eps, R_eps = sample_from_se3_gaussian(x, R, std)

    v_tar = se3_score_normal(x_eps, R_eps, x_tar=x, R_tar=R, std=std)

    v_pred = model(x_eps, R_eps, t)

    loss = (std.pow(2))*((v_pred - v_tar).pow(2).sum(-1))
    print(loss.mean())

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.mean().backward()
    optimizer.step()


## Test Model to Generate Samples ##

GENERATE_SAMPLES_DETERMINISTIC = True
if GENERATE_SAMPLES_DETERMINISTIC:

    R0 = SO3.rand(B).to_matrix()
    x0 = torch.randn(B, 3)

    T = 5000
    dt= 0.001
    for t in range(T):
        k = (T - t)/T + 10e-3

        v = model(x0, R0, k=k*torch.ones(x0.shape[0]))
        _s = v*dt
        x0, R0 = step(x0, R0, _s)

    xd, Rd = get_sample_from_data(10)

    print('X')
    print(x0)
    print('X_data')
    print(xd)

    print('R')
    print(R0)
    print('R_data')
    print(Rd)


GENERATE_SAMPLES_LANGEVIN = False
if GENERATE_SAMPLES_LANGEVIN:

    R0 = SO3.rand(B).to_matrix()
    x0 = torch.randn(B, 3)

    eps = 10e-2
    std_L = marginal_prob_std(t=torch.ones(1))


    T = 500
    for t in range(T):
        k = (T - t)/T + 10e-5

        v = model(x0, R0, k=k*torch.ones(x0.shape[0]))

        ## scale ##
        std_k = marginal_prob_std(k*torch.ones(x0.shape[0]))
        alpha = eps*std_k/std_L

        _s = v*alpha[:,None].pow(2) + alpha[:,None]*torch.randn_like(v)
        x0, R0 = step(x0, R0, _s)

    T = 500
    dt= 0.001
    for t in range(T):
        k = 10e-3

        v = model(x0, R0, k=k*torch.ones(x0.shape[0]))
        _s = v*dt
        x0, R0 = step(x0, R0, _s)

    xd, Rd = get_sample_from_data(10)

    print('X')
    print(x0)
    print('X_data')
    print(xd)

    print('R')
    print(R0)
    print('R_data')
    print(Rd)





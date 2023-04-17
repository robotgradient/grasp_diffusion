from isaacgym import gymapi,gymtorch
import torch
import numpy as np
from se3dif.utils import SO3_R3
import theseus as th


def H_2_Transform(H):
    so3r3_repr = SO3_R3(R=H[:3, :3].view(1,3,3), t=H[:3, -1].view(1,3))

    p = H[:3,-1]
    q = so3r3_repr.to_quaternion()[0,...]

    p = gymapi.Vec3(x=p[0], y=p[1], z=p[2])
    q = gymapi.Quat(w=q[0], x=q[1], y=q[2], z=q[3])

    return gymapi.Transform(p, q)

def Transform_2_H(T):
    H = torch.eye(4)
    # set position
    H[0, -1] = T.p.x
    H[1, -1] = T.p.y
    H[2, -1] = T.p.z
    # set rotation
    q = torch.Tensor([T.r.w, T.r.x, T.r.y, T.r.z])
    so3_repr = th.geometry.SO3(quaternion=q).to_matrix()
    H[:3,:3] = so3_repr
    return H

def pq_to_H(p, q):
    # expects as input: quaternion with convention [x y z w]
    # arrange quaternion with convention [w x y z] for theseus
    q = torch.Tensor([q[3],q [0], q[1], q[2]])
    so3_repr = th.geometry.SO3(quaternion=q).to_matrix()
    H = torch.eye(4).to(p)
    H[:3,:3] = so3_repr
    H[:3, -1] = p
    return H
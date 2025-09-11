import torch
import numpy as np
from collections import deque
from scipy.special import loggamma


def to_spherical(x):
    """
    Convert cartesian coordinates to spherical
    Input: x: torch.Tensor of size (..., D)
    Output: radii: corresponding radii of size (...)
            angles: corresponding angle vectors of size (..., D - 1)
    """
    denom = torch.flip(
        torch.cumsum(
            torch.flip(x.square(), dims=(-1, )), dim=-1
        ), dims=(-1, )
    ).sqrt()
    angles = torch.atan2(denom[..., 1:], x[..., :-1])
    angles[..., -1] = torch.atan2(x[..., -1], x[..., -2])
    radii = denom[..., 0]
    return radii, angles


class EntropyLogger(object):
    def __init__(self, queue_size, allow_cdist=False, thr=1e-8):
        self.queue_size = queue_size
        self.angles = deque(maxlen=queue_size)
        self.radii = deque(maxlen=queue_size)
        self.allow_cdist = allow_cdist
        self.thr = thr

    def add_weights(self, weights):
        radius, angle = to_spherical(weights)
        self.angles.append(angle)
        self.radii.append(radius.item())

    def get_radius(self):
        return torch.tensor(list(self.radii)).mean().item()

    def get_entropy(self):
        if len(self.angles) < self.queue_size:
            spherical_entropy, log_radius = np.nan, np.nan
        else:
            dim = self.angles[0].shape[0]
            N = self.queue_size

            if self.allow_cdist:
                angles = torch.stack(list(self.angles), dim=0)
                distances = torch.cdist(angles, angles).numpy()
            else:
                distances = np.zeros((N, N))
                for i in range(N):
                    for j in range(i):
                        distances[i, j] = torch.norm(self.angles[i] - self.angles[j]).item()
                distances = distances + distances.T

            knn_distances = np.sort(distances, axis=0)[1]
            knn_distances = np.clip(knn_distances, a_min=self.thr, a_max=None)
            angles_entropy = dim * np.log(knn_distances).mean() + dim / 2 * np.log(np.pi) - loggamma(dim / 2 + 1) + \
                         np.log(N - 1) + np.euler_gamma

            log_jacobian = 0.0
            mults = torch.arange(dim - 1, 0, -1)
            for i in range(N):
                log_jacobian += (
                    mults *
                    self.angles[i][:-1].sin().clip(min=self.thr).log()
                ).sum().item() / N

            spherical_entropy = angles_entropy + log_jacobian
            radius = self.get_radius()
            log_radius = dim * np.log(radius)

        return spherical_entropy, log_radius

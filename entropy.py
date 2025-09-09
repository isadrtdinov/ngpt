import torch
from spherical import to_spherical
from sklearn.neighbors import NearestNeighbors


def expected_log_jacobian(angles):
    assert angles.shape[-1] >= 3
    log_sines = angles[..., 1:-1].sin().log().mean(dim=-2)
    shape = (1, ) * (len(log_sines.shape) - 1) + (-1, )
    mults = torch.arange(angles.shape[-1] - 2, 0, -1).reshape(shape)
    log_jacobian = (log_sines * mults).sum(dim=-1)
    return log_jacobian


def entropy_kozachenko_leonenko(weights, mode='scale', effective_dim=None):
    assert mode in ['isotropic', 'scale', 'covariance']
    N, D = weights.shape[1:]
    dim = D if effective_dim is None else effective_dim
    entropy = []

    for weight in weights:
        normed_weight = weight - weight.mean(dim=0, keepdim=True)
        if mode == 'scale':
            std = normed_weight.std(dim=0, keepdim=True)
            normed_weight = normed_weight / std
        elif mode == 'covariance':
            sigma = normed_weight.T @ normed_weight / (N - 1)
            U, S, V = torch.svd(sigma)
            A = U * (1 / S.sqrt()) @ U.T
            normed_weight = weight @ A

        nb = NearestNeighbors(n_neighbors=2).fit(normed_weight)
        dists, _ = nb.kneighbors(normed_weight)
        value = dim * np.log(dists[..., 1]).mean() + dim / 2 * np.log(np.pi) - loggamma(dim / 2 + 1) + \
                np.log(N - 1) + np.euler_gamma

        if mode == 'scale':
            value += std.log().sum()
        elif mode == 'covariance':
            value += 0.5 * S.log().sum()
        entropy.append(value)

    return torch.tensor(entropy)


def spherical_entropy(weights):
    normalized_weights = weights / weights.norm(dim=-1, keepdim=True)
    angles = sp.to_spherical(normalized_weights)
    angles_ent = entropy_kozachenko_leonenko(angles[..., 1:])
    log_jac = expected_log_jacobian(angles)
    return angles_ent + log_jac
